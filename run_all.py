#!/usr/bin/env python3
"""一键复现：真实 fork 实测 → 事故复现 → 建议推演 → 参数网格 → 看板数据。

用法
----
  python3 run_all.py                # 全流程（网格约 10 分钟）
  python3 run_all.py --fast         # 快速档（小组合，约 1 分钟，用于自检）
  python3 run_all.py --skip-sweep   # 跳过最耗时的参数网格

产物（results/ 与 dashboard/）
-----------------------------
  results/validation.json   真实 fork 实测 vs 仿真误差
  results/incident.json     30/64GiB 事故复现
  results/recommend.json    五条建议与调整后推演
  results/sweep.json        参数网格（192 组或快速档）
  dashboard/report.json|js  看板数据源
  dashboard/index.html      浏览器打开即可（python3 -m http.server）
"""

from __future__ import annotations

import argparse
import json
import os
import time

from cowkit import real_fork
from cowkit.sweep import reproduce_incident, run_grid
from cowkit.recommend import evaluate
from cowkit.report import write_report

GIB = 1024 ** 3
RES = "results"


def step(title):
    print("\n" + "=" * 64 + f"\n{title}\n" + "=" * 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="快速自检档")
    ap.add_argument("--skip-sweep", action="store_true")
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    t0 = time.time()

    # 1) 真实 fork 实测（快速档用更小竞技场）
    step("① 真实 fork() 本机实测 vs 仿真")
    from cowkit.validate import main as validate_main
    validate_main(arena_mib=(64 if args.fast else 128),
                  writes=(4000 if args.fast else 12000),
                  out_path=f"{RES}/validation.json")

    # 2) 事故复现
    step("② 30GiB 数据 / 64GiB 机器事故复现")
    incident = reproduce_incident()
    json.dump(incident, open(f"{RES}/incident.json", "w"),
              ensure_ascii=False)
    for r in incident["runs"]:
        print(f"  [{'THP开' if r['huge_on'] else 'THP关'}] {r['workload']:<10} "
              f"峰值 {r['footprint_peak_bytes']/GIB:6.1f}GiB  "
              f"p99 {r['p99_us']:8.0f}us  OOM={r['oom']}")

    # 3) 建议
    step("③ 建议与调整后推演")
    rec = evaluate()
    json.dump(rec, open(f"{RES}/recommend.json", "w"),
              ensure_ascii=False, indent=2)
    for name, s in rec["scenarios"].items():
        print(f"  {name:<24} 峰值 {s['footprint_peak_bytes']/GIB:6.1f}GiB  "
              f"p99 {s['p99_us']:7.0f}us  OOM={s['oom']}")

    # 4) 参数网格
    if not args.skip_sweep:
        step("④ 参数网格推演")
        if args.fast:
            grid = run_grid(
                data_gib_list=(20, 30), rate_list=(4000,),
                snapshot_s_list=(60,),
                workloads=("uniform", "hotkey", "sequential", "mixed"),
                keep_curves_for=("uniform",))
        else:
            grid = run_grid()
        json.dump(grid, open(f"{RES}/sweep.json", "w"))
        n_oom = sum(1 for r in grid["rows"] if r["oom"])
        print(f"  {grid['meta']['n_combos']} 组，OOM {n_oom} 组，"
              f"耗时 {grid['meta']['elapsed_s']}s")

    # 5) 看板数据
    step("⑤ 生成看板数据")
    write_report()
    print(f"\n全部完成，总耗时 {time.time()-t0:.0f}s")
    print("打开看板： cd dashboard && python3 -m http.server 8765")
    print("          浏览器访问 http://localhost:8765/index.html")


if __name__ == "__main__":
    main()
