"""
生成前端 dashboard 所需的数据文件。

产物：
  web/data/dashboard.json   概览数字、场景矩阵、实测对比
  web/data/curves.json      每个场景的内存/复制曲线（页级仿真）
  web/data/heatmap.json     页热度图（按桶）
  web/data/layout.json      页布局示意（占用率、键大小、脏页数分布）
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from sim.layout import KVStore, SizeDistribution, layout_stats
from sim.model import (DIST_HOT, DIST_LABELS, DIST_SEQUENTIAL, DIST_UNIFORM,
                       DIST_ZIPF, Scenario, evaluate, resolve)
from sim.pagesim import BucketSim, run_sim
from sim.profiles import LINUX_4K

GIB = 1 << 30
MB = 1 << 20
OUT = os.path.join(ROOT, "web", "data")
os.makedirs(OUT, exist_ok=True)


# 用于曲线的场景（30 GiB 生产口径）
# (key, label, dist, huge, locality, snapshot_seconds, write_rate)
CURVE_SCENARIOS = [
    ("uniform", "均匀随机写", DIST_UNIFORM, False, 1.0, 120, 50_000),
    ("sequential", "顺序写（批量导入）", DIST_SEQUENTIAL, False, 1.0, 120, 50_000),
    ("hotkey", "少数热键反复写", DIST_HOT, False, 1.0, 120, 50_000),
    ("zipf", "Zipf 长尾写", DIST_ZIPF, False, 1.0, 120, 50_000),
    ("uniform-scattered", "均匀随机写（分配器打散热区）", DIST_UNIFORM, False, 0.3, 120, 50_000),
    ("uniform-huge", "均匀随机写 + 2MiB 大页", DIST_UNIFORM, True, 1.0, 120, 50_000),
    ("hotkey-huge", "热键写 + 2MiB 大页", DIST_HOT, True, 1.0, 120, 50_000),
    ("incident", "现场现状：30GiB·50k写·600s 慢快照", DIST_UNIFORM, False, 0.3, 600, 50_000),
    ("incident-thp", "现场现状变体：开 THP", DIST_UNIFORM, True, 0.3, 120, 50_000),
]


def downsample(arr: np.ndarray, n: int = 160) -> List[float]:
    if arr.size <= n:
        return [float(x) for x in arr]
    idx = np.linspace(0, arr.size - 1, n).astype(int)
    return [float(arr[i]) for i in idx]


def build_curves() -> Dict:
    out = {}
    for key, label, dist, huge, loc, snap_s, rate in CURVE_SCENARIOS:
        sc = Scenario(name=key, data_gib=30, util=0.85, write_rate=rate,
                      distribution=dist, locality=loc, snapshot_seconds=snap_s,
                      ram_gib=64, ncpu=8, huge_pages=huge)
        tr = BucketSim(sc, seed=1, n_samples=180).run()
        rv = resolve(sc, LINUX_4K)
        an = evaluate(sc, LINUX_4K, pressure=False)
        lat_t, lat_ms, lat_cow = tr.lat_t, tr.lat_ms, tr.lat_cow
        order = np.argsort(lat_t)
        out[key] = {
            "label": label,
            "distribution": dist,
            "huge": huge,
            "locality": loc,
            "snapshot_seconds": rv.snapshot_seconds,
            "resident_gib": rv.resident_bytes / GIB,
            "n_blocks": rv.n_blocks,
            "block_kib": rv.block_bytes // 1024,
            "t": downsample(tr.t),
            "copied": downsample(tr.copied_blocks),
            "mem_gib": downsample(tr.memory_bytes / GIB),
            "p99_ms": an["latency"]["p99_ms"],
            "p999_ms": an["latency"]["p999_ms"],
            "peak_gib": tr.memory_bytes.max() / GIB,
            "peak_ratio": tr.memory_bytes.max() / tr.resident_bytes,
            "copied_fraction": float(tr.copied_blocks[-1] / tr.n_blocks),
            "rho": tr.memory_bytes.max() / GIB / 64.0,
            # 延迟采样：按时间排序，用于与复制事件对齐
            "lat": [{"t": float(lat_t[i]), "ms": float(lat_ms[i]),
                     "cow": bool(lat_cow[i])}
                    for i in order[:: max(1, order.size // 400)]],
        }
    return out


def build_heatmaps() -> Dict:
    """每种分布：桶索引 -> 归一化热度 + 累计复制比例随时间。"""
    out = {}
    for key, label, dist, huge, loc, snap_s, rate in CURVE_SCENARIOS:
        sc = Scenario(name=key, data_gib=30, util=0.85, write_rate=rate,
                      distribution=dist, locality=loc, snapshot_seconds=snap_s,
                      ram_gib=64, huge_pages=huge)
        nb = 4096
        tr = BucketSim(sc, seed=2, n_samples=180).run()
        sizes, mass = tr.heat if tr.heat is not None else (np.array([1.0]), np.array([1.0]))
        # 按秩重采样到固定 256 列
        total = float(np.sum(sizes))
        if total <= 0:
            heat = np.zeros(256)
        else:
            cum = np.concatenate(([0.0], np.cumsum(sizes) / total))
            grid = np.linspace(0, 1, 257)
            idx = np.searchsorted(cum, grid, side="right") - 1
            idx = np.clip(idx, 0, len(mass) - 1)
            heat = mass[idx][:256]
            heat = heat / max(1e-12, heat.max())
        out[key] = {
            "label": label,
            "heat": [round(float(x), 5) for x in heat],
            "huge": huge,
        }
    return out


def build_layout() -> Dict:
    """小规模精确布局：60 万键、4 KiB 页，画出每页占用率与脏页数分布。"""
    rng = np.random.default_rng(11)
    sd = SizeDistribution(median_small=96, sigma_small=0.9, p_large=0.004,
                          large_alpha=1.6, large_cap=2 * MB)
    sizes = sd.sample(600_000, rng)
    st = KVStore(4096, rng)
    st.bulk_insert(sizes)

    # 制造删除/过期留下的空洞
    victims = rng.choice(np.flatnonzero(st.key_start >= 0), 120_000, replace=False)
    st.delete(victims)
    new_sizes = sd.sample(90_000, rng)
    st.insert(new_sizes)

    stats = layout_stats(st)
    page_frac = st.page_used / st.page_size
    # 每页一个样本画热力条（最多 512 列，按页号取均值）
    ncols = 512
    edges = np.linspace(0, st.n_pages, ncols + 1).astype(int)
    col_occ = np.array([page_frac[edges[i]:edges[i + 1]].mean()
                        if edges[i + 1] > edges[i] else 0.0 for i in range(ncols)])
    col_live = np.array([st.page_live[edges[i]:edges[i + 1]].sum()
                         for i in range(ncols)])

    # 写一个键弄脏多少页（按写次数加权，用抽样近似）
    ids = rng.integers(0, st.n_keys, 200_000)
    live = st.key_start[ids] >= 0
    dirty = st.key_npages[ids][live]
    hist, bins = np.histogram(dirty, bins=[1, 2, 3, 4, 5, 8, 16, 32, 64, 128, 1024])
    dirty_hist = {f"{int(bins[i])}-{int(bins[i+1]-1)}": int(hist[i]) for i in range(len(hist))}

    size_hist, size_bins = np.histogram(
        sizes, bins=[8, 32, 64, 128, 256, 1024, 4096, 16384, 65536, 262144, 2 * MB])

    return {
        "page_size": 4096,
        "n_keys": int(st.n_keys),
        "n_pages": int(st.n_pages),
        "data_mib": st.data_bytes() / MB,
        "used_mib": st.used_bytes / MB,
        "util": st.util,
        "stats": stats.as_dict(),
        "occupancy": st.occupancy_hist(),
        "col_occ": [round(float(x), 4) for x in col_occ],
        "col_live": [int(x) for x in col_live],
        "dirty_hist": dirty_hist,
        "size_hist": {f"{int(size_bins[i])}-{int(size_bins[i+1])}": int(size_hist[i])
                      for i in range(len(size_hist))},
    }


def build_matrix() -> Dict:
    """预计算所有参数组合，避免前端近似重算与模型漂移。"""
    out = []
    for data_gib in (10, 20, 30, 40):
        for snap_s in (60, 120, 300):
            for huge in (False, True):
                for dist in (DIST_UNIFORM, DIST_HOT, DIST_ZIPF, DIST_SEQUENTIAL):
                    for rate in (5_000, 25_000, 50_000, 100_000, 200_000):
                        sc = Scenario(name="mx", data_gib=data_gib, util=0.85,
                                      write_rate=rate, distribution=dist,
                                      locality=0.7, snapshot_seconds=snap_s,
                                      ram_gib=64, ncpu=8, huge_pages=huge)
                        r = evaluate(sc, LINUX_4K, pressure=False)
                        out.append({"data_gib": data_gib, "snapshot_seconds": snap_s,
                                    "huge": huge, "distribution": dist,
                                    "write_rate": rate,
                                    "peak_gib": r["peak_gib"],
                                    "peak_ratio": r["peak_ratio"],
                                    "rho": r["rho"]})
    return {"rows": out}


def main() -> int:
    curves = build_curves()
    heat = build_heatmaps()
    layout = build_layout()

    with open(os.path.join(OUT, "curves.json"), "w") as fh:
        json.dump(curves, fh, ensure_ascii=False)
    with open(os.path.join(OUT, "heatmap.json"), "w") as fh:
        json.dump(heat, fh, ensure_ascii=False)
    with open(os.path.join(OUT, "layout.json"), "w") as fh:
        json.dump(layout, fh, ensure_ascii=False)
    with open(os.path.join(OUT, "matrix.json"), "w") as fh:
        json.dump(build_matrix(), fh, ensure_ascii=False)

    # dashboard 汇总：复用 reports/summary.json
    summary_path = os.path.join(ROOT, "reports", "summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as fh:
            summary = json.load(fh)
    with open(os.path.join(OUT, "dashboard.json"), "w") as fh:
        json.dump({
            "baseline": summary.get("baseline"),
            "recommendations": summary.get("recommendations"),
            "bench_vs_model": summary.get("bench_vs_model"),
            "sim_vs_model": summary.get("sim_vs_model"),
            "volume_sweep": summary.get("volume_sweep"),
            "scenario_matrix": summary.get("scenario_matrix"),
            "profiles": summary.get("profiles"),
            "measured": summary.get("bench"),
        }, fh, ensure_ascii=False)
    print("curves", list(curves))
    print("heatmaps", list(heat))
    print("layout pages", layout["n_pages"], "util", round(layout["util"], 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
