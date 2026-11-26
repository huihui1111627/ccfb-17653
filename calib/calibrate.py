"""
本机标定：macOS / arm64 / 16 KiB 页。

测量项
  1. fork 停顿与数据区大小的关系（COW 标记要在 fork 里做）
  2. 父进程写一页触发 COW 的单次停顿时延分布（p50/p90/p99/p99.9）
  3. 无 COW 的普通写延迟（对照）
  4. 不同大小 memcpy 带宽（内核复制页的近似成本上界）
  5. 子进程顺序扫描吞吐（快照序列化的 CPU 上限）
  6. Python 堆 vs 裸 mmap 在 fork 后的额外 COW（说明为什么实测要用裸 mmap）

输出：calib/measured.json
"""
import ctypes
import ctypes.util
import json
import mmap
import os
import platform
import re
import statistics
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MB = 2 ** 20
GIB = 2 ** 30
PAGE = mmap.PAGESIZE

libc = ctypes.CDLL(ctypes.util.find_library("c"))
libc.mach_task_self.restype = ctypes.c_uint

TASK_EVENTS_INFO = 2


class TaskEventsInfo(ctypes.Structure):
    _fields_ = [
        ("faults", ctypes.c_int32),
        ("pageins", ctypes.c_int32),
        ("cow_faults", ctypes.c_int32),
        ("messages_sent", ctypes.c_int32),
        ("messages_received", ctypes.c_int32),
        ("syscalls_mach", ctypes.c_int32),
        ("syscalls_unix", ctypes.c_int32),
        ("csw", ctypes.c_int32),
    ]


def cow_faults():
    info = TaskEventsInfo()
    count = ctypes.c_uint32(ctypes.sizeof(TaskEventsInfo) // 4)
    kr = libc.task_info(libc.mach_task_self(), TASK_EVENTS_INFO,
                        ctypes.byref(info), ctypes.byref(count))
    if kr != 0:
        raise OSError(f"task_info failed: {kr}")
    return info.cow_faults


def vm_stat():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    stats = {}
    for line in out.splitlines():
        m = re.match(r'"?([^:"]+)"?:\s+(\d+)', line.strip())
        if m:
            stats[m.group(1).strip()] = int(m.group(2))
    return stats


def anon_mib():
    return vm_stat()["Anonymous pages"] * PAGE / MB


def quantiles_us(lat_ns):
    a = np.asarray(lat_ns, dtype=np.float64) / 1e3
    if a.size == 0:
        return {}
    q = np.percentile(a, [50, 90, 99, 99.9, 100])
    return {
        "n": int(a.size),
        "p50_us": round(float(q[0]), 3),
        "p90_us": round(float(q[1]), 3),
        "p99_us": round(float(q[2]), 3),
        "p999_us": round(float(q[3]), 3),
        "max_us": round(float(q[4]), 3),
        "mean_us": round(float(a.mean()), 3),
    }


def fork_latency(gib, reps=7):
    n = int(gib * GIB)
    buf = mmap.mmap(-1, n, flags=mmap.MAP_PRIVATE | mmap.MAP_ANON)
    for off in range(0, n, PAGE):
        buf[off] = 1
    time.sleep(0.05)
    lats = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        dt = time.perf_counter_ns() - t0
        os.waitpid(pid, 0)
        lats.append(dt / 1e6)
    buf.close()
    return {
        "arena_gib": gib,
        "pages": n // PAGE,
        "p50_ms": round(statistics.median(lats), 3),
        "min_ms": round(min(lats), 3),
        "max_ms": round(max(lats), 3),
    }


def cow_write_latency(mib, reps_pages=None, sleep_child=2.0):
    """父进程逐页写，子进程存活；测得每次首写的 COW 停顿。"""
    n = mib * MB
    buf = mmap.mmap(-1, n, flags=mmap.MAP_PRIVATE | mmap.MAP_ANON)
    for off in range(0, n, PAGE):
        buf[off] = 1
    time.sleep(0.05)
    before = cow_faults()
    pid = os.fork()
    if pid == 0:
        time.sleep(sleep_child)
        os._exit(0)
    time.sleep(0.05)
    lats = []
    limit = n if reps_pages is None else min(n, reps_pages * PAGE)
    for off in range(0, limit, PAGE):
        t0 = time.perf_counter_ns()
        buf[off] = 2
        lats.append(time.perf_counter_ns() - t0)
    faults = cow_faults() - before
    os.waitpid(pid, 0)
    buf.close()
    res = quantiles_us(lats)
    res["cow_faults"] = int(faults)
    res["pages_written"] = limit // PAGE
    return res


def plain_write_latency(mib):
    n = mib * MB
    buf = mmap.mmap(-1, n, flags=mmap.MAP_PRIVATE | mmap.MAP_ANON)
    for off in range(0, n, PAGE):
        buf[off] = 1
    time.sleep(0.05)
    lats = []
    for off in range(0, n, PAGE):
        t0 = time.perf_counter_ns()
        buf[off] = 3
        lats.append(time.perf_counter_ns() - t0)
    buf.close()
    return quantiles_us(lats)


def memcpy_bw(size_bytes, reps):
    src = np.ones(size_bytes // 8, dtype=np.float64)
    dst = np.empty_like(src)
    lats = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        np.copyto(dst, src)
        lats.append(time.perf_counter_ns() - t0)
    q = quantiles_us(lats)
    med_s = q["p50_us"] / 1e6
    return {
        "block_kib": size_bytes // 1024,
        "p50_us": q["p50_us"],
        "p99_us": q["p99_us"],
        "bandwidth_gib_s": round(size_bytes / med_s / GIB, 2),
    }


def child_scan_throughput(mib):
    n = mib * MB
    buf = mmap.mmap(-1, n, flags=mmap.MAP_PRIVATE | mmap.MAP_ANON)
    for off in range(0, n, PAGE):
        buf[off] = 1
    time.sleep(0.05)
    pid = os.fork()
    if pid == 0:
        t0 = time.perf_counter()
        acc = 0
        for off in range(0, n, PAGE):
            acc += buf[off]
        dt = time.perf_counter() - t0
        with open("/tmp/.scan_result", "w") as fh:
            fh.write(f"{dt} {acc}")
        os._exit(0)
    os.waitpid(pid, 0)
    buf.close()
    dt, _ = open("/tmp/.scan_result").read().split()
    dt = float(dt)
    return {
        "scan_mib": mib,
        "seconds": round(dt, 4),
        "throughput_gib_s": round(mib / 1024 / dt, 2),
        "pages_per_s": int(mib * MB / PAGE / dt),
    }


def python_vs_raw_cow():
    """同样读 200 MiB，Python 对象 vs 裸 mmap，子进程侧的 COW 页数。"""
    n_keys = 100_000
    blob = bytes(512)
    store = [blob] * n_keys
    arena = mmap.mmap(-1, n_keys * 512, flags=mmap.MAP_PRIVATE | mmap.MAP_ANON)
    view = memoryview(arena)
    for i in range(n_keys):
        view[i * 512:(i + 1) * 512] = blob
    time.sleep(0.1)

    def measure(fn):
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(r)
            time.sleep(0.1)
            a = cow_faults()
            checksum = fn()
            b = cow_faults()
            os.write(w, f"{b - a},{checksum}".encode())
            os.close(w)
            os._exit(0)
        os.close(w)
        out = os.read(r, 128).decode()
        os.waitpid(pid, 0)
        return int(out.split(",")[0])

    py = measure(lambda: sum(v[0] for v in store))
    raw = measure(lambda: sum(view[i * 512] for i in range(n_keys)))
    del view
    arena.close()
    return {"touched_bytes_mib": n_keys * 512 // MB,
            "python_obj_child_cow_pages": py,
            "raw_mmap_child_cow_pages": raw}


def main():
    result = {
        "platform": {
            "system": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "page_size_bytes": PAGE,
            "physical_mib": int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                               capture_output=True, text=True).stdout.strip()) // MB,
        },
        "fork": [fork_latency(g) for g in (0.25, 0.5, 1, 2, 4, 6)],
        "cow_write_16k": cow_write_latency(256),
        "plain_write_16k": plain_write_latency(256),
        "memcpy": [memcpy_bw(4096, 4000), memcpy_bw(16384, 3000),
                   memcpy_bw(2 * MB, 400)],
        "child_scan": child_scan_throughput(1024),
        "python_vs_raw_cow": python_vs_raw_cow(),
    }
    with open(os.path.join(HERE, "measured.json"), "w") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
