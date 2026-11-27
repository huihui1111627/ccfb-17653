"""参数组合推演：数据量 × 写入速率 × 大页开关 × 写入模式。

输出每个组合的：
  峰值 footprint、复制页数、OOM 与否、p50/p99/p999/p9999 延迟、fork 停顿。
大场景（30 GiB 数据 / 64 GiB 内存）用 PageArena 逐页模型，
一个组合只保留汇总指标与降采样曲线，避免内存爆炸。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict

import numpy as np

from .pages import ArenaConfig, PageArena, THP_2M
from .workloads import WorkloadSpec
from .engine import simulate_snapshot, LatencyModel
from .analytic import AnalyticInput, peak_memory

GIB = 1024 ** 3


def run_grid(data_gib_list=(10, 20, 30, 40),
             rate_list=(1000, 4000, 10000),
             snapshot_s_list=(60, 180),
             workloads=("uniform", "hotkey", "sequential", "mixed"),
             huge_options=(0, THP_2M),
             ram_gib: float = 64.0,
             page_size: int = 4096,
             fill_fraction: float = 0.85,
             latency: LatencyModel | None = None,
             keep_curves_for: tuple = ("uniform",),
             seed: int = 11) -> dict:
    """跑完整网格。曲线只对 keep_curves_for 里的模式保留（控制 JSON 体积）。"""
    latency = latency or LatencyModel()
    ram = int(ram_gib * GIB)
    rows = []
    curves = {}
    t_start = time.time()

    for data_gib in data_gib_list:
        for huge in huge_options:
            cfg = ArenaConfig(
                data_bytes=int(data_gib * GIB), page_size=page_size,
                huge_size=huge, fill_fraction=fill_fraction, seed=seed)
            arena = PageArena(cfg)
            for wl_name in workloads:
                avg_q = 1.0 if wl_name != "mixed" else (0.92 * 1 + 0.08 * 16)
                for rate in rate_list:
                    for dur in snapshot_s_list:
                        spec = WorkloadSpec(wl_name, write_rate=rate,
                                            duration=dur, seed=seed)
                        res = simulate_snapshot(
                            arena, spec, ram_bytes=ram,
                            model=latency, n_buckets=90,
                            heatmap_cells=1500)
                        summ = res.summary()
                        # 闭式公式交叉校验
                        # 闭式公式交叉校验：huge_fill 用竞技场真实已映射占比
                        # （活页+空洞），而不是 fill_fraction（仅活页占比）
                        actual_fill = arena.n_mapped / len(arena.state)
                        ai = AnalyticInput(
                            n_mapped_pages=arena.n_mapped, page_size=page_size,
                            writes_total=spec.n_writes, workload=wl_name,
                            huge_size=huge, avg_pages_per_write=avg_q,
                            huge_fill=actual_fill)
                        formula = peak_memory(ai, arena.n_mapped * page_size)
                        summ.update({
                            "data_gib": data_gib,
                            "ram_gib": ram_gib,
                            "rate": rate,
                            "snapshot_s": dur,
                            "huge_on": bool(huge),
                            "fill_fraction": fill_fraction,
                            "formula_copied_pages": formula["copied_pages_peak"],
                            "formula_sim_err_pct":
                                100.0 * abs(formula["copied_pages_peak"]
                                            - summ["copied_pages_peak"])
                                / max(1, summ["copied_pages_peak"]),
                        })
                        rows.append(summ)
                        if wl_name in keep_curves_for and dur == max(snapshot_s_list):
                            key = f"{wl_name}|{data_gib}GiB|rate{rate}|thp{int(bool(huge))}"
                            curves[key] = {
                                "t": res.t.tolist(),
                                "copied_gib": (res.copied_pages * page_size / GIB).tolist(),
                                "footprint_gib": (res.footprint_bytes / GIB).tolist(),
                                "writes": res.writes_cum.tolist(),
                                "huge_on": bool(huge), "data_gib": data_gib,
                                "rate": rate, "workload": wl_name,
                                "oom_at": res.oom_at,
                            }
            del arena

    return {
        "meta": {"ram_gib": ram_gib, "page_size": page_size,
                 "fill_fraction": fill_fraction,
                 "elapsed_s": round(time.time() - t_start, 1),
                 "n_combos": len(rows)},
        "rows": rows,
        "curves": curves,
    }


def reproduce_incident(ram_gib: float = 64.0, data_gib: float = 30.0,
                       rate: int = 4000, snapshot_s: float = 180,
                       latency: LatencyModel | None = None) -> dict:
    """复现用户的事故：30 GiB 数据 / 64 GiB 机器 / THP 开，快照几分钟冲到 ~60 GiB。"""
    latency = latency or LatencyModel()
    ram = int(ram_gib * GIB)
    out = {"params": {"ram_gib": ram_gib, "data_gib": data_gib,
                      "rate": rate, "snapshot_s": snapshot_s}, "runs": []}
    for huge in (0, THP_2M):
        tag = "THP开(2MiB大页)" if huge else "THP关(4KiB基础页)"
        cfg = ArenaConfig(data_bytes=int(data_gib * GIB), page_size=4096,
                          huge_size=huge, fill_fraction=0.85, seed=21)
        arena = PageArena(cfg)
        for wl_name in ("uniform", "hotkey", "sequential", "mixed"):
            spec = WorkloadSpec(wl_name, rate, snapshot_s, seed=21)
            res = simulate_snapshot(arena, spec, ram, model=latency,
                                    n_buckets=120, heatmap_cells=2400)
            s = res.summary()
            s["tag"] = tag
            s["workload"] = wl_name
            out["runs"].append(s)
            if wl_name == "uniform":
                out.setdefault("curves", {})[tag] = {
                    "t": res.t.tolist(),
                    "copied_gib": (res.copied_pages * 4096 / GIB).tolist(),
                    "footprint_gib": (res.footprint_bytes / GIB).tolist(),
                    "oom_at": res.oom_at}
        del arena
    return out


if __name__ == "__main__":
    import sys
    out_path = sys.argv[1] if len(sys.argv) > 1 else "results/sweep.json"
    report = run_grid()
    with open(out_path, "w") as f:
        json.dump(report, f)
    print(f"combos={report['meta']['n_combos']} elapsed={report['meta']['elapsed_s']}s")
    print("->", out_path)
