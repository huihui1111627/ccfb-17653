"""真的 fork() 派生子进程的本机 COW 实测。

流程
====
1. mmap 一块匿名内存（默认 256 MiB），写入随机非零字节 —— 强制物理帧真正分配
   （全零页会被内核压成零页单例，测不出真实复制）；
2. 记录 fork 前父进程的次缺页计数与物理 footprint；
3. os.fork()：子进程持有同一映射，睡眠等待；父子此刻共享全部物理页；
4. 父进程按三种写入模式触碰页，逐次计时：
   - 第一次写某页 => COW 缺页 => 内核复制整页，计入 getrusage 的 ru_minflt
     （次缺页）；用写循环前后的 minflt 差值得到内核级精确复制页数；
   - 再写同一页 => 不再故障 => 对照样本；
   单页首次写与重复写的耗时差，就是缺页复制停顿的本机实测值。
5. 父进程通知子进程退出，输出 JSON，供和仿真结果对比误差。

为什么不用信号 / footprint 差值
==============================
* 现代 macOS 的 COW 故障在内核静默修复，不向用户态投递 SIGBUS/SIGSEGV；
* macOS 内存压缩器会很快回收 COW 产生的匿名页，phys_footprint 差值严重偏小；
* getrusage(RUSAGE_SELF).ru_minflt 是内核维护的计数，macOS/Linux 均可移植，
  COW 复制正是“次缺页（minor fault）”的典型来源。

注意
====
* macOS（本机）没有面向用户的透明大页，THP 只能在仿真里测；这里测基础页 COW。
* 全程不超过 arena 的 2 倍内存，默认 256 MiB，16 GiB 机器安全。
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import mmap
import os
import resource
import signal
import sys
import time

import numpy as np

LIBPROC = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)


def minor_faults() -> int:
    """进程累计次缺页数（COW 复制、按需调零页都算 minor fault）。"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_minflt


# ---------------------------------------------------------------------- #
# footprint 读取：macOS 走 Mach task_info(TASK_VM_INFO)，Linux 走 /proc
# ---------------------------------------------------------------------- {
class _TaskVmInfo(ctypes.Structure):
    _fields_ = [
        ("phys_footprint", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("resident_size_peak", ctypes.c_uint64),
        ("device", ctypes.c_uint64),
        ("page_size", ctypes.c_uint64),
        ("free", ctypes.c_uint64 * 14),
    ]


def mach_phys_footprint() -> int:
    info = _TaskVmInfo()
    count = ctypes.c_uint32(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_int))
    TASK_VM_INFO = 22
    rc = LIBPROC.task_info(
        LIBPROC.mach_task_self(), TASK_VM_INFO,
        ctypes.byref(info), ctypes.byref(count))
    if rc != 0:
        raise OSError(ctypes.get_errno(), "task_info failed")
    return int(info.phys_footprint)


def linux_rss() -> int:
    with open("/proc/self/statm") as f:
        pages = int(f.read().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE")


def read_footprint() -> int:
    if sys.platform == "darwin":
        return mach_phys_footprint()
    return linux_rss()


def native_page_size() -> int:
    if sys.platform == "darwin":
        import subprocess
        return int(subprocess.check_output(["sysctl", "-n", "hw.pagesize"]))
    return os.sysconf("SC_PAGE_SIZE")


# ---------------------------------------------------------------------- #
def run_once(arena_bytes: int, mode: str, n_writes: int, seed: int) -> dict:
    page = native_page_size()
    rng = np.random.default_rng(seed)

    # 1) 建竞技场并强制物理驻留
    mm = mmap.mmap(-1, arena_bytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANON,
                   prot=mmap.PROT_READ | mmap.PROT_WRITE)
    buf = np.frombuffer(mm, dtype=np.uint8)
    buf[:] = rng.integers(1, 255, size=arena_bytes, dtype=np.uint8)
    n_pages = arena_bytes // page
    time.sleep(0.2)
    fp_before = read_footprint()

    # 2) fork：子进程持映射睡眠
    pid = os.fork()
    if pid == 0:
        # 子进程：什么都不写，等父进程结束自己
        try:
            for _ in range(600):
                time.sleep(0.1)
        finally:
            os._exit(0)

    # 父进程：稳定后记录缺页基线（fork 之后，子进程不共享 rusage 计数）
    time.sleep(0.1)
    fp_after_fork = read_footprint()
    minflt_base = minor_faults()

    # 3) 按模式生成目标页号
    if mode == "uniform":
        targets = rng.integers(0, n_pages, size=n_writes)
    elif mode == "hotkey":
        n_hot = max(1, n_pages // 20)
        hot = rng.integers(0, n_hot, size=int(n_writes * 0.8))
        cold = rng.integers(0, n_pages, size=n_writes - int(n_writes * 0.8))
        targets = np.concatenate([hot, cold])
        rng.shuffle(targets)
    elif mode == "sequential":
        targets = (np.arange(n_writes) % n_pages).astype(np.int64)
    else:
        raise ValueError(mode)
    target_list = [int(x) for x in targets.tolist()]

    # 4) 逐页写并计时：首次写某页 => 内核 COW 次缺页，重复写为对照
    seen = np.zeros(n_pages, dtype=bool)
    fault_us, repeat_us = [], []
    t0 = time.perf_counter()
    for pno in target_list:
        off = int(pno) * page + int(rng.integers(0, max(1, page - 1)))
        first_touch = not seen[pno]
        t1 = time.perf_counter_ns()
        mm[off] = (mm[off] + 1) & 0xFF          # 单字节写 => 触发/不触发 COW
        dt = (time.perf_counter_ns() - t1) / 1000.0
        if first_touch:
            seen[pno] = True
            fault_us.append(dt)                  # 首次写：含缺页复制停顿
        else:
            repeat_us.append(dt)                 # 重复写：纯对照
    wall_s = time.perf_counter() - t0

    # 内核级精确 COW 复制页数 = 写循环产生的次缺页
    kernel_fault_pages = minor_faults() - minflt_base
    time.sleep(0.2)
    fp_end = read_footprint()
    unique_pages = int(seen.sum())
    expected_copied = unique_pages * page
    measured_copied = max(0, fp_end - fp_after_fork)

    os.kill(pid, signal.SIGTERM)
    os.waitpid(pid, 0)
    del buf
    mm.close()

    def pct(a, q):
        return float(np.percentile(a, q)) if a else float("nan")

    return {
        "platform": sys.platform,
        "mode": mode,
        "page_size": page,
        "arena_bytes": arena_bytes,
        "arena_pages": n_pages,
        "n_writes": int(n_writes),
        "unique_pages_touched": unique_pages,
        "target_pages": target_list,
        "kernel_cow_faults": kernel_fault_pages,
        "fault_vs_touched_error_pct":
            100.0 * abs(kernel_fault_pages - unique_pages) / max(1, unique_pages),
        "expected_copied_bytes": expected_copied,
        "footprint_before_fork": fp_before,
        "footprint_after_fork": fp_after_fork,
        "footprint_end": fp_end,
        "footprint_delta_bytes": measured_copied,
        "footprint_delta_error_pct":
            100.0 * abs(measured_copied - expected_copied) / max(1, expected_copied),
        "fault_us": {
            "p50": pct(fault_us, 50), "p99": pct(fault_us, 99),
            "max": max(fault_us) if fault_us else float("nan"),
            "n": len(fault_us)},
        "repeat_write_us": {
            "p50": pct(repeat_us, 50), "p99": pct(repeat_us, 99),
            "n": len(repeat_us)},
        "wall_s": wall_s,
        "note": "复制页数以 getrusage 次缺页(minflt)差值为准（内核精确计数）；"
                "footprint 差值受 macOS 内存压缩影响，仅作辅助。本机无用户态 THP",
    }


def main():
    ap = argparse.ArgumentParser(description="真实 fork() COW 本机实测")
    ap.add_argument("--arena-mib", type=int, default=256)
    ap.add_argument("--writes", type=int, default=20000)
    ap.add_argument("--modes", nargs="+",
                    default=["uniform", "hotkey", "sequential"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    arena = args.arena_mib * 1024 * 1024
    results = []
    for mode in args.modes:
        r = run_once(arena, mode, args.writes, args.seed)
        results.append(r)
        print(f"[{mode:>10s}] 触碰页={r['unique_pages_touched']:>6d}  "
              f"内核故障={r['kernel_cow_faults']:>6d}  "
              f"预期复制={r['expected_copied_bytes']/1e6:8.1f}MB  "
              f"footprintΔ={r['footprint_delta_bytes']/1e6:7.1f}MB(受压缩)  "
              f"故障写 p50={r['fault_us']['p50']:7.2f}us "
              f"p99={r['fault_us']['p99']:8.2f}us  "
              f"重复写 p50={r['repeat_write_us']['p50']:6.2f}us")

    payload = {"arena_mib": args.arena_mib, "writes": args.writes,
               "results": results}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print("written:", args.out)


if __name__ == "__main__":
    main()
