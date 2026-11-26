"""
生成分析报告：解析模型 vs 页级仿真 vs 真机实测，三方对比并给结论。

输出：
  reports/summary.json        全部数值
  reports/report.md           人读的长文
  web/data/dashboard.json     页面数据
"""
from __future__ import annotations

import glob
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from sim.model import (DIST_HOT, DIST_LABELS, DIST_SEQUENTIAL, DIST_UNIFORM,
                       DIST_ZIPF, Scenario, evaluate, resolve)
from sim.pagesim import BucketSim, BitSetSim, run_sim
from sim.profiles import DARWIN_16K, DARWIN_16K_HUGE_EMU, LINUX_4K, PROFILES, get
from sim.heat import groups_for

GIB = 1 << 30
MB = 1 << 20


# ---------------------------------------------------------------------------
# 1. 现场问题：30 GiB / 64 GiB / 每天快照
# ---------------------------------------------------------------------------
def incident_baseline() -> Dict:
    """
    复现现场：30 GiB 数据、64 GiB 内存、每天定时快照。

    "每天那几分钟内存涨到近 60 GiB、两次 OOM、延迟几十毫秒" —— 用模型算，
    需要多长的快照窗口、多高的写入速率才会撞到 2 倍上界。
    """
    rows = []
    for rate in (5_000, 20_000, 50_000, 100_000, 200_000):
        for snap_s in (30, 60, 120, 300, 600):
            sc = Scenario(name=f"r{rate}_t{snap_s}", data_gib=30, util=0.85,
                          write_rate=rate, distribution=DIST_UNIFORM,
                          snapshot_seconds=snap_s, ram_gib=64, ncpu=8)
            r = evaluate(sc, LINUX_4K, pressure=False)
            rows.append({
                "write_rate": rate, "snapshot_seconds": snap_s,
                "writes_per_page": r["writes_per_page"],
                "peak_gib": r["peak_gib"], "peak_ratio": r["peak_ratio"],
                "copied_gib": r["copied_gib"], "rho": r["rho"],
                "oom_risk": r["peak_gib"] >= 0.97 * 64,
            })
    # 现场"延迟几十毫秒"需要的条件：快照窗口内写入足够多，让 85% 水线被越过
    lat_rows = []
    for rate in (10_000, 50_000, 100_000, 200_000):
        for snap_s in (60, 180, 600):
            sc = Scenario(name=f"lat_{rate}_{snap_s}", data_gib=30, util=0.85,
                          write_rate=rate, distribution=DIST_UNIFORM,
                          snapshot_seconds=snap_s, ram_gib=64, ncpu=8)
            r = evaluate(sc, LINUX_4K, pressure=False)
            lat_rows.append({
                "write_rate": rate, "snapshot_seconds": snap_s,
                "peak_gib": r["peak_gib"],
                "p99_ms": r["latency"]["p99_ms"],
                "p999_ms": r["latency"]["p999_ms"],
                "pressure_ms": r["pressure_stall_peak_ms"],
            })
    return {"peak_table": rows, "latency_table": lat_rows}


# ---------------------------------------------------------------------------
# 2. 写入模式 × 数据量 × 速率的推演矩阵
# ---------------------------------------------------------------------------
def scenario_matrix(data_gib: float = 30.0, ram_gib: float = 64.0,
                    snapshot_seconds: float = 120.0) -> List[Dict]:
    rows = []
    for dist in (DIST_UNIFORM, DIST_HOT, DIST_ZIPF, DIST_SEQUENTIAL):
        for rate in (5_000, 25_000, 50_000, 100_000, 200_000):
            for loc in (1.0, 0.3):
                for huge in (False, True):
                    sc = Scenario(name=f"{dist}_{rate}_{loc}_{huge}", data_gib=data_gib,
                                  util=0.85, write_rate=rate, distribution=dist,
                                  locality=loc, snapshot_seconds=snapshot_seconds,
                                  ram_gib=ram_gib, ncpu=8, huge_pages=huge)
                    r = evaluate(sc, LINUX_4K, pressure=False)
                    rows.append({
                        "distribution": dist,
                        "distribution_label": DIST_LABELS[dist],
                        "locality": loc,
                        "write_rate": rate,
                        "huge": huge,
                        "peak_gib": r["peak_gib"],
                        "peak_ratio": r["peak_ratio"],
                        "copied_fraction": r["copied_fraction"],
                        "writes_per_page": r["writes_per_page"],
                        "p99_ms": r["latency"]["p99_ms"],
                        "p999_ms": r["latency"]["p999_ms"],
                        "cow_cpu": r["latency"]["cow_cpu_cores_peak"],
                        "rho": r["rho"],
                        "oom_risk": r["oom_risk"],
                    })
    return rows


def data_volume_sweep() -> List[Dict]:
    """单机放多少数据安全：不同数据量、快照 60s、写 50k/s。"""
    rows = []
    for data_gib in (5, 10, 15, 20, 25, 30, 35, 40, 50):
        for dist in (DIST_UNIFORM, DIST_HOT):
            sc = Scenario(name=f"vol_{data_gib}_{dist}", data_gib=data_gib,
                          util=0.85, write_rate=50_000, distribution=dist,
                          snapshot_seconds=60, ram_gib=64, ncpu=8)
            r = evaluate(sc, LINUX_4K, pressure=False)
            rows.append({"data_gib": data_gib, "distribution": dist,
                         "peak_gib": r["peak_gib"], "peak_ratio": r["peak_ratio"],
                         "rho": r["rho"],
                         "headroom_gib": 64 - r["peak_gib"],
                         "oom_risk": r["oom_risk"]})
    return rows


# ---------------------------------------------------------------------------
# 3. 建议方案的效果
# ---------------------------------------------------------------------------
def recommendation_effects(snapshot_seconds: float = 120.0) -> List[Dict]:
    """
    对比四种调整：
      现状        30 GiB, 全量 COW 快照, THP 开
      方案 A       数据压到 20 GiB
      方案 B       快照窗口缩短（增量 dump / 记录快照点后 fork 立即返回）
      方案 C       关掉 THP + 数据重排（提升 locality）+ 控制在 22 GiB
    """
    out = []

    def one(name, data_gib, huge, loc, snap_s, dist=DIST_UNIFORM):
        sc = Scenario(name=name, data_gib=data_gib, util=0.85,
                      write_rate=50_000, distribution=dist, locality=loc,
                      snapshot_seconds=snap_s, ram_gib=64, ncpu=8,
                      huge_pages=huge)
        r = evaluate(sc, LINUX_4K, pressure=False)
        return {"name": name, "data_gib": data_gib, "huge": huge,
                "locality": loc, "snapshot_seconds": snap_s,
                "peak_gib": r["peak_gib"], "peak_ratio": r["peak_ratio"],
                "rho": r["rho"], "p99_ms": r["latency"]["p99_ms"],
                "p999_ms": r["latency"]["p999_ms"],
                "cow_cpu": r["latency"]["cow_cpu_cores_peak"],
                "oom_risk": r["oom_risk"]}

    out.append(one("现状 30GiB+THP+慢快照", 30, True, 0.3, snapshot_seconds))
    out.append(one("A 只把数据压到20GiB", 20, True, 0.3, snapshot_seconds))
    out.append(one("B 只缩短快照到60s", 30, True, 0.3, 60))
    out.append(one("C 只关THP+数据重排", 30, False, 0.9, snapshot_seconds))
    out.append(one("D A+B 压量+快照加速", 20, True, 0.3, 60))
    out.append(one("E A+C 压量+关THP", 20, False, 0.9, snapshot_seconds))
    out.append(one("F A+B+C 全上", 20, False, 0.9, 60))
    return out


# ---------------------------------------------------------------------------
# 4. 真机实测汇总
# ---------------------------------------------------------------------------
def load_bench() -> List[Dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "bench", "results", "*.json"))):
        with open(path) as fh:
            r = json.load(fh)
        if "cow_pages" not in r:
            continue
        cfg = r["config"]
        rows.append({
            "file": os.path.basename(path),
            "name": cfg["name"],
            "data_gib": r["resident_gib"],
            "distribution": cfg["distribution"],
            "huge": cfg["huge"],
            "write_rate": cfg["write_rate"],
            "duration_s": r["snapshot_seconds"],
            "n_pages": r["n_pages"],
            "cow_pages": r["cow_pages"],
            "peak_gib": r["peak_gib"],
            "peak_ratio": r["peak_ratio"],
            "cow_fraction": r["cow_fraction"],
            "lat_cow_p50_us": r.get("latency_cow", {}).get("p50_us"),
            "lat_cow_p99_us": r.get("latency_cow", {}).get("p99_us"),
            "lat_plain_p50_us": r.get("latency_plain", {}).get("p50_us"),
            "lat_plain_p99_us": r.get("latency_plain", {}).get("p99_us"),
            "sys_anon_delta_mib": r.get("system_internal_delta_mib"),
            "fork_ms": r.get("fork_ms"),
            "curve": r.get("curve"),
        })
    return rows


# ---------------------------------------------------------------------------
# 4b. 实测 vs 推演
# ---------------------------------------------------------------------------
BENCH_TO_MODEL = {"uniform": "uniform", "sequential": "sequential",
                  "hot": "hotkey", "zipf": "zipf", "mixed": "uniform"}


def bench_vs_model() -> List[Dict]:
    """把每条真机实测对应的场景喂给模型，比较峰值倍数与复制比例。

    实测 arena 是等长灌满的（util=1.0），对比时必须用同一口径，
    否则 0.85 的利用率会把页数放大 18%，凭空造出 18% 的误差。
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "bench", "results", "*.json"))):
        with open(path) as fh:
            r = json.load(fh)
        if "cow_pages" not in r:
            continue
        cfg = r["config"]
        prof = DARWIN_16K_HUGE_EMU if cfg["huge"] else DARWIN_16K
        n_pages = r["n_pages"]
        data_gib = n_pages * prof.base_page / GIB
        sc = Scenario(name=cfg["name"], data_gib=data_gib, util=1.0,
                      write_rate=r["actual_rate"],
                      distribution=BENCH_TO_MODEL[cfg["distribution"]],
                      snapshot_seconds=r["snapshot_seconds"],
                      profile=prof.key, huge_pages=cfg["huge"],
                      reclaim_ms_at_100pct=0.0)
        m = evaluate(sc, prof, pressure=False)
        rv = resolve(sc, prof)
        model_copied = m["copied_fraction"] * rv.n_blocks
        rows.append({
            "name": cfg["name"],
            "group": r.get("group", "memory"),
            "distribution": cfg["distribution"],
            "huge": cfg["huge"],
            "data_gib": r["resident_gib"],
            "measured_peak_ratio": r["peak_ratio"],
            "model_peak_ratio": m["peak_ratio"],
            "peak_err_pct": abs(m["peak_ratio"] - r["peak_ratio"]) / r["peak_ratio"] * 100,
            "measured_copied": r["cow_pages"],
            "model_copied": model_copied,
            "copied_err_pct": (abs(model_copied - r["cow_pages"]) / r["cow_pages"] * 100
                               if cfg["huge"] is False else None),
            "measured_cow_p50_us": r.get("latency_cow", {}).get("p50_us"),
            "measured_cow_p99_us": r.get("latency_cow", {}).get("p99_us"),
            "model_cow_p50_us": rv.t_cow_us_p50,
            "model_cow_p99_us": rv.t_cow_us_p99,
        })
    return rows


# ---------------------------------------------------------------------------
# 5. 解析 vs 页级仿真的对照实验
# ---------------------------------------------------------------------------
def sim_vs_model(data_gib: float = 4.0) -> List[Dict]:
    rows = []
    for dist in (DIST_UNIFORM, DIST_HOT, DIST_ZIPF, DIST_SEQUENTIAL):
        for loc in (1.0, 0.3):
            for huge in (False, True):
                sc = Scenario(name=f"{dist}_{loc}_{huge}", data_gib=data_gib, util=0.85,
                              write_rate=50_000, distribution=dist, locality=loc,
                              snapshot_seconds=30, ram_gib=64, ncpu=8,
                              huge_pages=huge)
                an = evaluate(sc, LINUX_4K, pressure=False)
                sim = run_sim(sc).summary()
                rows.append({
                    "distribution": dist, "locality": loc, "huge": huge,
                    "model_peak_gib": an["peak_gib"],
                    "sim_peak_gib": sim["peak_gib"],
                    "rel_err_pct": abs(an["peak_gib"] - sim["peak_gib"]) / max(1e-9, an["peak_gib"]) * 100,
                    "model_copied_frac": an["copied_fraction"],
                    "sim_copied_frac": sim["copied_fraction"],
                })
    return rows


def main() -> int:
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "web", "data"), exist_ok=True)

    summary = {
        "baseline": incident_baseline(),
        "scenario_matrix": scenario_matrix(),
        "volume_sweep": data_volume_sweep(),
        "recommendations": recommendation_effects(),
        "bench": load_bench(),
        "bench_vs_model": bench_vs_model(),
        "sim_vs_model": sim_vs_model(),
        "profiles": {k: v.as_dict() for k, v in PROFILES.items()},
    }
    with open(os.path.join(ROOT, "reports", "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
    print("baseline rows", len(summary["baseline"]["peak_table"]))
    print("scenario rows", len(summary["scenario_matrix"]))
    print("bench rows", len(summary["bench"]))
    bvm = [r for r in summary["bench_vs_model"] if r["group"] == "memory"]
    print("bench_vs_model rows", len(bvm))
    for r in bvm:
        print(f"  {r['name']:28s} peak meas={r['measured_peak_ratio']:.3f} "
              f"model={r['model_peak_ratio']:.3f} err={r['peak_err_pct']:5.2f}%")
    print("sim_vs_model rows", len(summary["sim_vs_model"]))
    for r in summary["sim_vs_model"]:
        print(f"  {r['distribution']:10s} loc={r['locality']} huge={r['huge']} "
              f"model={r['model_peak_gib']:.3f} sim={r['sim_peak_gib']:.3f} err={r['rel_err_pct']:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
