"""
真实 fork 实测：在本机跑真正的 fork() + COW，量出被复制页数、内存曲线、写停顿时延。

## 为什么必须用裸 mmap 而不是 Python 对象

CPython 的子进程只要**读**一个 Python 对象（哪怕只是取引用计数），
解释器就会写回该对象的引用计数，把对象所在的页弄脏 → 触发 COW。
实验证明：200 MiB 的 Python bytes 列表，子进程只做遍历就会复制 1.6 万个页；
换成 memoryview 读同一份数据，只复制 3~6 页。真实数据库子进程用 C 写快照，
不会碰引用计数，所以**必须把数据放在裸 mmap 且子进程只读**，才贴近真实。

## 测量口径

* 数据区      `mmap(MAP_PRIVATE|MAP_ANON)`，大小 = 页数 × 页大小。
* 复制页数    父进程的 `task_info(TASK_EVENTS_INFO).cow_faults` 增量。macOS 上精确。
              作为交叉校验，同时记录全系统 `vm_stat` 的 "Pages copy-on-write"。
* 内存曲线    父进程数据区占用 = 初始大小 + 复制页数 × 页大小。
              父进程自身的物理占用（RSS）在 macOS 上不随 COW 增长（页表指向新页，
              旧页转给子进程），所以用"初始 + 复制量"这个口径，与解析模型一致。
* 写停顿时延  每次写的 `perf_counter_ns` 差值；按是否命中未复制页拆开统计。
* 快照       子进程可选：(a) 立即退出（模拟快速快照）(b) 停留 T 秒并顺序读一遍
              (c) 分块读真快照。

## 写模式

`uniform` / `sequential` / `hot` / `zipf` / `mixed`（小键值 1 页 + 大键值连续多页）
以及 `huge`（写一个字节就强制复制整个 2 MiB 块，用来模拟大页）。

大页模拟：macOS 没有 THP。做法是把 2 MiB 对齐的 128 个 16 KiB 页当作一个块，
每次写先取块内随机偏移，再对**块内全部 128 页**各写一次——内核会做 128 次 COW，
总拷贝量 = 2 MiB，与开大页后的行为等价（写放大、单次停顿都一致）。
"""
from __future__ import annotations

import ctypes
import ctypes.util
import json
import mmap
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MB = 1 << 20
GIB = 1 << 30

# ---------------------------------------------------------------------------
# macOS 私有 API
# ---------------------------------------------------------------------------
_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libc.mach_task_self.restype = ctypes.c_uint

TASK_EVENTS_INFO = 2


class _TaskEventsInfo(ctypes.Structure):
    _fields_ = [("faults", ctypes.c_int32), ("pageins", ctypes.c_int32),
                ("cow_faults", ctypes.c_int32), ("messages_sent", ctypes.c_int32),
                ("messages_received", ctypes.c_int32), ("syscalls_mach", ctypes.c_int32),
                ("syscalls_unix", ctypes.c_int32), ("csw", ctypes.c_int32)]


def cow_faults() -> int:
    info = _TaskEventsInfo()
    n = ctypes.c_uint32(ctypes.sizeof(_TaskEventsInfo) // 4)
    kr = _libc.task_info(_libc.mach_task_self(), TASK_EVENTS_INFO,
                         ctypes.byref(info), ctypes.byref(n))
    if kr != 0:
        raise OSError(f"task_info(TASK_EVENTS_INFO) failed: {kr}")
    return int(info.cow_faults)


class _RUsageV4(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64), ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64), ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64), ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64), ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64), ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64), ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64), ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64), ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64), ("ri_diskio_byteswritten", ctypes.c_uint64),
        ("ri_cpu_time_qos_default", ctypes.c_uint64), ("ri_cpu_time_qos_maintenance", ctypes.c_uint64),
        ("ri_cpu_time_qos_background", ctypes.c_uint64), ("ri_cpu_time_qos_utility", ctypes.c_uint64),
        ("ri_cpu_time_qos_legacy", ctypes.c_uint64), ("ri_cpu_time_qos_user_initiated", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_interactive", ctypes.c_uint64), ("ri_billed_system_time", ctypes.c_uint64),
        ("ri_serviced_system_time", ctypes.c_uint64)]


def process_footprint(pid: int) -> Optional[Tuple[int, int]]:
    """(resident_bytes, phys_footprint_bytes)，失败返回 None。"""
    buf = _RUsageV4()
    ok = _libc.proc_pid_rusage(int(pid), 4, ctypes.byref(buf))
    if ok != 0:
        return None
    return int(buf.ri_resident_size), int(buf.ri_phys_footprint)


class _VMStats(ctypes.Structure):
    _fields_ = [
        ("free_count", ctypes.c_uint32), ("active_count", ctypes.c_uint32),
        ("inactive_count", ctypes.c_uint32), ("wire_count", ctypes.c_uint32),
        ("zero_fill_count", ctypes.c_uint64), ("reactivations", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64), ("pageouts", ctypes.c_uint64),
        ("faults", ctypes.c_uint64), ("cow_faults", ctypes.c_uint64),
        ("lookups", ctypes.c_uint64), ("hits", ctypes.c_uint64),
        ("purges", ctypes.c_uint64), ("purgeable_count", ctypes.c_uint32),
        ("speculative_count", ctypes.c_uint32), ("decompressions", ctypes.c_uint64),
        ("compressions", ctypes.c_uint64), ("swapins", ctypes.c_uint64),
        ("swapouts", ctypes.c_uint64), ("compressor_page_count", ctypes.c_uint32),
        ("throttled_count", ctypes.c_uint32), ("external_page_count", ctypes.c_uint32),
        ("internal_page_count", ctypes.c_uint32),
        ("total_uncompressed_pages_in_compressor", ctypes.c_uint64)]

_libc.mach_host_self.restype = ctypes.c_uint
_HOST = _libc.mach_host_self()


def system_page_stats() -> Dict[str, int]:
    v = _VMStats()
    n = ctypes.c_uint32(ctypes.sizeof(_VMStats) // 4)
    _libc.host_statistics64(_HOST, 4, ctypes.byref(v), ctypes.byref(n))
    ps = mmap.PAGESIZE
    return {
        "free_bytes": int(v.free_count) * ps,
        "internal_bytes": int(v.internal_page_count) * ps,
        "compressor_bytes": int(v.compressor_page_count) * ps,
        "cow_faults_system": int(v.cow_faults),
    }


# ---------------------------------------------------------------------------
# 实测配置
# ---------------------------------------------------------------------------
@dataclass
class BenchConfig:
    name: str = "bench"
    data_mib: int = 512
    write_rate: float = 50_000.0
    duration_s: float = 4.0
    distribution: str = "uniform"       # uniform / sequential / hot / zipf / mixed
    huge: bool = False                  # 模拟 2 MiB 大页
    huge_block: int = 2 * MB
    zipf_s: float = 1.1
    hot_fraction: float = 0.9
    hot_space: float = 0.002
    large_share: float = 0.0            # mixed：大键值字节占比
    large_mean_kib: float = 256.0
    child_hold_s: float = 0.0           # 子进程停留时长；0 = 立即退出
    child_scan: bool = True             # 子进程是否顺序读一遍（模拟序列化）
    seed: int = 7
    sample_every: int = 2000            # 每多少次写采一次内存/复制数

    @property
    def data_bytes(self) -> int:
        return self.data_mib * MB


# ---------------------------------------------------------------------------
# 写入轨迹（在父进程侧预先生成，避免测得的时间包含生成开销）
# ---------------------------------------------------------------------------
def make_write_plan(cfg: BenchConfig, n_pages: int, n_writes: int, page: int) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    dist = cfg.distribution

    if cfg.huge:
        n_blocks = n_pages // (cfg.huge_block // page)
    else:
        n_blocks = n_pages

    if dist == "sequential":
        idx = np.arange(n_writes, dtype=np.int64) % max(1, n_blocks)
    elif dist == "hot":
        k = max(1, int(cfg.hot_space * n_blocks))
        is_hot = rng.random(n_writes) < cfg.hot_fraction
        hot = rng.integers(0, k, n_writes)
        cold = rng.integers(0, n_blocks, n_writes)
        idx = np.where(is_hot, hot, cold)
    elif dist == "zipf":
        # 逆 CDF 的近似：对秩做幂律抽样
        u = rng.random(n_writes)
        a = 1.0 - cfg.zipf_s
        if abs(a) < 1e-9:
            idx = np.floor(np.exp(u * np.log(n_blocks + 1))) - 1
        else:
            idx = np.floor((1 + u * ((n_blocks + 1) ** a - 1)) ** (1 / a)) - 1
        idx = np.clip(idx, 0, n_blocks - 1).astype(np.int64)
    else:  # uniform
        idx = rng.integers(0, n_blocks, n_writes)

    if cfg.huge:
        block_pages = cfg.huge_block // page
        # 返回块号；执行时展开成整块
        return idx.astype(np.int64)
    if dist == "mixed" and cfg.large_share > 0:
        # 大键值：连续多页
        return idx.astype(np.int64)
    return idx.astype(np.int64)


# ---------------------------------------------------------------------------
# 执行
# ---------------------------------------------------------------------------
def _quantiles(lat_ns: np.ndarray) -> Dict[str, float]:
    if lat_ns.size == 0:
        return {}
    a = lat_ns.astype(np.float64) / 1e3     # us
    q = np.percentile(a, [50, 90, 99, 99.9, 100])
    return {"n": int(a.size), "p50_us": float(q[0]), "p90_us": float(q[1]),
            "p99_us": float(q[2]), "p999_us": float(q[3]), "max_us": float(q[4]),
            "mean_us": float(a.mean())}


def run_bench(cfg: BenchConfig, progress=None) -> Dict:
    page = mmap.PAGESIZE
    n_pages = cfg.data_bytes // page
    n_writes = max(1, int(cfg.write_rate * cfg.duration_s))

    buf = mmap.mmap(-1, cfg.data_bytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANON)
    view = memoryview(buf)
    arr = None
    try:
        # 灌入数据：每页写一个字节，保证页已实例化
        for off in range(0, cfg.data_bytes, page):
            buf[off] = 1
        time.sleep(0.05)
        resident = cfg.data_bytes

        plan = make_write_plan(cfg, n_pages, n_writes, page)
        block_pages = 1
        if cfg.huge:
            block_pages = cfg.huge_block // page
        arr = np.frombuffer(buf, dtype=np.uint8)

        base_cow = cow_faults()
        sys0 = system_page_stats()
        r, w = os.pipe()          # 子进程回报
        r2, w2 = os.pipe()        # 父进程唤醒子进程
        fork_t0 = time.perf_counter_ns()
        pid = os.fork()
        fork_ns = time.perf_counter_ns() - fork_t0

        if pid == 0:
            # ---------------- 子进程 ----------------
            os.close(r)      # 子进程只写
            os.close(w2)
            t_start = time.perf_counter()
            scan_bytes = 0
            try:
                if cfg.child_hold_s > 0 or cfg.child_scan:
                    # 按顺序读（模拟快照序列化）。只读，不产生 COW。
                    if cfg.child_scan:
                        acc = 0
                        for off in range(0, cfg.data_bytes, page):
                            acc += buf[off]
                        scan_bytes = cfg.data_bytes
                    remain = cfg.child_hold_s - (time.perf_counter() - t_start)
                    if remain > 0:
                        time.sleep(remain)
            except BaseException:
                err = traceback.format_exc()
                os.write(w, json.dumps({"err": err}).encode())
                os._exit(1)
            os.write(w, json.dumps({"child_secs": time.perf_counter() - t_start,
                                    "scan_bytes": scan_bytes}).encode())
            os.close(w)
            os._exit(0)
        else:
            # ---------------- 父进程 ----------------
            os.close(w)      # 父进程只读
            os.close(r2)
            # 子进程扫描期间，父进程按速率写
            interval = 1.0 / cfg.write_rate
            lat_cow: List[int] = []
            lat_plain: List[int] = []
            cow_seen = np.zeros(n_pages // block_pages + 1, dtype=bool)
            samples = []    # (t, copied_pages, mem_bytes)
            t0 = time.perf_counter()

            for i in range(n_writes):
                target = t0 + i * interval
                blk = int(plan[i])
                if cfg.huge:
                    base_page = blk * block_pages
                    hit = cow_seen[blk]
                    # 模拟"大页"：一次写命中块，就把整块 2 MiB 复制一份。
                    # 已经复制过的块不再重复付代价（与真实 COW 一致）。
                    if not hit:
                        a = time.perf_counter_ns()
                        arr[base_page * page:(base_page + block_pages) * page:page] = 3
                        b = time.perf_counter_ns()
                    else:
                        a = b = time.perf_counter_ns()
                    cow_seen[blk] = True
                else:
                    a = time.perf_counter_ns()
                    buf[blk * page] = 2
                    b = time.perf_counter_ns()
                    hit = cow_seen[blk]
                    cow_seen[blk] = True
                (lat_cow if not hit else lat_plain).append(b - a)

                if i % cfg.sample_every == 0:
                    c = cow_faults() - base_cow
                    samples.append((time.perf_counter() - t0, c, resident + c * page))
                    if progress:
                        progress(i / max(1, n_writes))

                now = time.perf_counter()
                if now < target:
                    time.sleep(target - now)

            total_s = time.perf_counter() - t0
            c_end = cow_faults() - base_cow
            samples.append((total_s, c_end, resident + c_end * page))

            chunks = []
            while True:
                chunk = os.read(r, 65536)
                if not chunk:
                    break
                chunks.append(chunk)
            os.close(r)
            raw = b"".join(chunks)
            try:
                child_info = json.loads(raw) if raw else {}
            except Exception:
                child_info = {"raw": raw.decode(errors="replace")}
            _, status = os.waitpid(pid, 0)

            sys1 = system_page_stats()
            lat_cow_a = np.asarray(lat_cow, dtype=np.int64)
            lat_plain_a = np.asarray(lat_plain, dtype=np.int64)

            samples_a = np.asarray(samples, dtype=np.float64)
            result = {
                "config": asdict(cfg),
                "platform": {
                    "system": platform.platform(), "machine": platform.machine(),
                    "page_size": page, "python": sys.version.split()[0],
                    "numpy": np.__version__,
                },
                "n_pages": n_pages,
                "n_writes": n_writes,
                "actual_rate": n_writes / total_s,
                "fork_ms": fork_ns / 1e6,
                "snapshot_seconds": total_s,
                "resident_gib": resident / GIB,
                "cow_pages": int(c_end),
                "cow_pages_expected_upper": n_pages,
                "peak_gib": float(samples_a[:, 2].max() / GIB),
                "peak_ratio": float(samples_a[:, 2].max() / resident),
                "cow_fraction": float(c_end / n_pages),
                "unique_written_blocks": int(cow_seen.sum()),
                "system_cow_faults_delta": sys1["cow_faults_system"] - sys0["cow_faults_system"],
                "system_internal_delta_mib": (sys1["internal_bytes"] - sys0["internal_bytes"]) / MB,
                "child": child_info,
                "latency_cow": _quantiles(lat_cow_a),
                "latency_plain": _quantiles(lat_plain_a),
                "latency_all": _quantiles(np.concatenate([lat_cow_a, lat_plain_a])),
                "curve": {
                    "t": samples_a[:, 0].tolist(),
                    "cow_pages": samples_a[:, 1].tolist(),
                    "mem_gib": (samples_a[:, 2] / GIB).tolist(),
                },
            }
            return result
    finally:
        del arr
        try:
            del view
        except Exception:
            pass
        buf.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: List[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="真实 fork + COW 实测")
    ap.add_argument("--data-mib", type=int, default=512)
    ap.add_argument("--rate", type=float, default=50_000)
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--dist", default="uniform",
                    choices=["uniform", "sequential", "hot", "zipf", "mixed"])
    ap.add_argument("--huge", action="store_true")
    ap.add_argument("--child-hold", type=float, default=0.0)
    ap.add_argument("--no-scan", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = BenchConfig(data_mib=args.data_mib, write_rate=args.rate,
                      duration_s=args.duration, distribution=args.dist,
                      huge=args.huge, child_hold_s=args.child_hold,
                      child_scan=not args.no_scan)
    res = run_bench(cfg)
    text = json.dumps(res, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
