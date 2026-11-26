"""
跑实测矩阵并落盘。

两组实验：
  A. 内存曲线矩阵：数据量 × 分布 × 大页 × 快照时长，子进程持留到写结束，
     目的是让 COW 窗口完整打开，测峰值内存与复制页数。
  B. 延迟矩阵：短时刻测量（大页 vs 基页），看单次写的停顿时延分布。

输出：bench/results/*.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import bench.fork_bench as fb

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)


def run_memory_matrix() -> List[Dict]:
    cfgs = []
    for data_mib in (128, 256, 512):
        for dist in ("uniform", "sequential", "hot", "zipf"):
            cfgs.append(fb.BenchConfig(
                name=f"mem_{data_mib}mib_{dist}", data_mib=data_mib,
                write_rate=40_000, duration_s=4.0, distribution=dist,
                child_hold_s=4.0, child_scan=False))
    # 大页：同样的写次数，块数少 128 倍
    for data_mib in (128, 512):
        cfgs.append(fb.BenchConfig(
            name=f"mem_{data_mib}mib_uniform_huge", data_mib=data_mib,
            write_rate=40_000, duration_s=4.0, distribution="uniform",
            huge=True, child_hold_s=4.0, child_scan=False))

    out = []
    for i, cfg in enumerate(cfgs):
        print(f"[{i+1}/{len(cfgs)}] {cfg.name} ...", flush=True)
        try:
            r = fb.run_bench(cfg)
        except Exception as exc:
            r = {"config": asdict(cfg), "error": repr(exc)}
        print(f"    cow={r.get('cow_pages')} peak={r.get('peak_gib'):.3f}GiB "
              f"ratio={r.get('peak_ratio'):.3f}", flush=True)
        r["group"] = "memory"
        out.append(r)
        with open(os.path.join(OUT, f"{cfg.name}.json"), "w") as fh:
            json.dump(r, fh, indent=1)
    return out


def run_latency_matrix() -> List[Dict]:
    cfgs = []
    for huge in (False, True):
        for dist in ("uniform", "hot"):
            cfgs.append(fb.BenchConfig(
                name=f"lat_{'huge' if huge else 'base'}_{dist}", data_mib=256,
                write_rate=20_000, duration_s=2.0, distribution=dist,
                huge=huge, child_hold_s=2.0, child_scan=False, sample_every=500))
    out = []
    for i, cfg in enumerate(cfgs):
        print(f"[lat {i+1}/{len(cfgs)}] {cfg.name} ...", flush=True)
        r = fb.run_bench(cfg)
        r["group"] = "latency"
        out.append(r)
        with open(os.path.join(OUT, f"{cfg.name}.json"), "w") as fh:
            json.dump(r, fh, indent=1)
        lc = r.get("latency_cow", {})
        print(f"    COW p50={lc.get('p50_us')}us p99={lc.get('p99_us')}us n={lc.get('n')}",
              flush=True)
    return out


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if which in ("all", "memory"):
        run_memory_matrix()
    if which in ("all", "latency"):
        run_latency_matrix()
    print(f"done in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
