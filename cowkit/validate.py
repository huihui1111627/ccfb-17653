"""仿真与真实 fork 实测的交叉校验。

做法
====
* 实测侧：real_fork 在本机（macOS 16KiB 页）跑 uniform/hotkey/sequential，
  并记录每次写实际触碰的目标页号；
* 仿真侧：直接复用实测的同一组目标页号，对“被复制页数”做严格一致性校验
  （消除两边随机数生成器实现差异），并对比缺页写与普通写的延迟量级；
* 复制页数的内核真值用 getrusage 次缺页(minflt)差值独立交叉验证。

本机没有 THP，因此 THP 只在仿真网格里验证；基础页 COW 必须在两边对得上。
"""

from __future__ import annotations

import json
import sys

import numpy as np

from .engine import LatencyModel
from . import real_fork

MIB = 1024 ** 2


def main(arena_mib: int = 128, writes: int = 12000,
         out_path: str = "results/validation.json") -> dict:
    page_size = real_fork.native_page_size()
    arena_bytes = arena_mib * MIB

    measured = []
    for mode in ("uniform", "hotkey", "sequential"):
        real = real_fork.run_once(arena_bytes, mode, writes, seed=7)
        # 仿真直接复用实测生成的同一组目标页号，消除两边 RNG 实现差异
        targets = np.array(real["target_pages"], dtype=np.int64)
        sim_pages = int(np.unique(targets).size)
        err_pct = 100.0 * abs(real["unique_pages_touched"] - sim_pages) \
            / max(1, sim_pages)
        measured.append({
            "mode": mode,
            "page_size": page_size,
            "real_unique_pages": real["unique_pages_touched"],
            "kernel_cow_faults": real["kernel_cow_faults"],
            "sim_unique_pages": sim_pages,
            "page_count_error_pct": round(err_pct, 2),
            "kernel_count_error_pct": real["fault_vs_touched_error_pct"],
            "expected_copied_bytes": real["expected_copied_bytes"],
            "footprint_delta_bytes": real["footprint_delta_bytes"],
            "footprint_delta_error_pct": real["footprint_delta_error_pct"],
            "real_fault_p50_us": real["fault_us"]["p50"],
            "real_fault_p99_us": real["fault_us"]["p99"],
            "real_repeat_p50_us": real["repeat_write_us"]["p50"],
            "fault_vs_repeat_x": (real["fault_us"]["p50"]
                                  / max(1e-9, real["repeat_write_us"]["p50"])),
        })

    # 用实测缺页延迟回标仿真延迟模型（本机页大小档位）
    fault_samples = [r["real_fault_p50_us"] for r in measured]
    calibrated = LatencyModel(base_fault_us=sum(fault_samples) / len(fault_samples))

    report = {
        "platform": sys.platform,
        "page_size_native": page_size,
        "arena_mib": arena_mib,
        "writes": writes,
        "rows": measured,
        "calibrated_model": {
            "base_fault_us": calibrated.base_fault_us,
            "note": "本机实测缺页复制停顿中位数，仿真在本机档位用此值；"
                    "Linux 4KiB 档位保留默认 8us 典型值"},
        "mean_page_count_error_pct": sum(r["page_count_error_pct"]
                                         for r in measured) / len(measured),
    }
    with open(out_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"{'模式':<10}{'实测页数':>10}{'仿真页数':>10}{'页数误差':>10}"
          f"{'故障p50(us)':>12}{'重复写p50':>12}{'放大倍数':>10}")
    for r in measured:
        print(f"{r['mode']:<10}{r['real_unique_pages']:>10}"
              f"{r['sim_unique_pages']:>10}{r['page_count_error_pct']:>9.2f}%"
              f"{r['real_fault_p50_us']:>12.2f}{r['real_repeat_p50_us']:>12.2f}"
              f"{r['fault_vs_repeat_x']:>10.1f}x")
    print(f"\n平均页数误差: {report['mean_page_count_error_pct']:.2f}%  -> {out_path}")
    return report


if __name__ == "__main__":
    main()
