"""
COW 峰值内存与延迟的解析模型。

## 一、基本画面

fork() 之后父子共享所有私有匿名页，页表项标成只读。父进程第一次写某页时
触发缺页，内核分配新页、拷贝旧内容、改页表——**每个页在一次快照里最多被复制一次**。
所以

    峰值内存 = 快照开始时的物理页占用 + 快照期间被复制过的页数 × 页大小

"快照开始时的物理页占用"就是数据常驻内存（含空洞、含 slab 内部碎片），
它由有效利用率 u 决定：`P = ceil(数据量 / (页大小 × u))`。
注意 P 用的是**物理页**，不是逻辑字节数——空洞和碎片一样要占内存、一样要被复制。

于是峰值倍数 = 1 + E/P，E 是快照结束时累计复制的页数，E ≤ P。
**最坏情况就是 2 倍**：所有页都被写过一次。30 GB 涨到 60 GB 正是撞上了这个上界。

## 二、复制曲线

父进程以速率 λ（次/秒）写。第 i 个页被单次写命中的概率记 w_i（Σw_i = 1）。
n 次写之后

    E(n) = Σ_i [ 1 - (1 - w_i)^n ]

- 均匀随机写：w_i = 1/P → `E(n) = P (1 - e^{-n/P})`（指数饱和，先快后慢）
- 顺序写（批量导入）：`E(n) = min(n, P)`（严格线性，直到写满）
- 热键：少数页承担大部分写 → E 很早就饱和，峰值接近 1 倍
- Zipf：介于两者之间，前 x% 的页承担大部分写，但尾部很长

快照时长 T_s 决定 n = λ·T_s。**峰值只取决于 n/P 这个比值**，也就是
"快照窗口内平均每个页被写了几次"。这是全文最重要的一条。

## 三、大页

开了 2 MiB 透明大页后，命中一个**块**就复制整个 2 MiB。复制单元数变成 P/512，
但每复制一次是 2 MiB：

    E_blocks(n) = B (1 - e^{-n/B}),  B = P/512
    复制字节 = E_blocks × 2 MiB

- 曲线形状不变，但横轴缩短 512 倍 → 极小的写入量就把数据复制光；
- 单次写的停顿从 ~1.3 µs 变成 ~45 µs（Linux），p99 更是几十到上百 µs；
- 父进程花在 COW 上的 CPU = λ·(dE/dn)·t_cow，50k/s × 45 µs ≈ 2.25 核，直接打满。

## 四、延迟

一次请求遇到 COW 的概率 q(t) 取决于它写的那几个块里有没有还没复制的。
请求延迟 = 基线 + (命中 COW ? 一次复制停顿 : 0)。

所以 p99 是否被拖高，取决于 q 是否大于 0.01；p99.9 取决于 q 是否大于 0.001。
快照前期 q ≈ 1（几乎每写必复制），后期趋于 0。**快照刚开始的那几十秒最痛。**

## 五、耦合（死亡螺旋）

峰值内存逼近物理内存 → 内存压缩/换页介入 → 子进程写快照变慢 → T_s 变长
→ n 变大 → 更多页被复制 → 峰值更高。这是一个正反馈。
模型里用 `pressure` 迭代求不动点，并标出是否发散。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .profiles import PageProfile

MB = 1 << 20
GIB = 1 << 30

DIST_UNIFORM = "uniform"
DIST_ZIPF = "zipf"
DIST_SEQUENTIAL = "sequential"
DIST_HOT = "hotkey"

DIST_LABELS = {
    DIST_UNIFORM: "均匀随机写",
    DIST_ZIPF: "Zipf 长尾写",
    DIST_SEQUENTIAL: "顺序写（批量导入）",
    DIST_HOT: "少数热键反复写",
}


@dataclass
class Scenario:
    name: str = "scenario"
    data_gib: float = 30.0
    util: float = 0.85
    write_rate: float = 50_000.0
    distribution: str = DIST_UNIFORM
    zipf_s: float = 1.1
    hot_fraction: float = 0.9        # 写入中落在热区的比例
    hot_space: float = 0.002         # 热区占页空间的比例（0.2%）
    locality: float = 1.0            # 1=页级热度按业务分布；0=被分配器打散成均匀
    large_share: float = 0.0         # 大键值字节占比
    large_mean_kib: float = 256.0
    small_mean_bytes: float = 120.0
    snapshot_mb_s: float = 1500.0
    snapshot_seconds: Optional[float] = None
    profile: str = "linux-4k"
    huge_pages: bool = False
    ram_gib: float = 64.0
    ncpu: int = 8
    concurrency: int = 32
    baseline_p50_ms: float = 1.6
    baseline_p99_ms: float = 8.0
    baseline_p999_ms: float = 20.0
    write_amplification: float = 1.0  # 一次逻辑写触发的物理写次数（RocksDB WAL+memtable 等）
    reclaim_ms_at_100pct: float = 35.0  # 内存打满时的直接回收/规整停顿（毫秒）

    def resolved(self, prof: PageProfile) -> "Resolved":
        return resolve(self, prof)


@dataclass
class Resolved:
    page_size: int
    block_bytes: int
    pages_per_block: int
    n_pages: int
    n_blocks: int
    data_bytes: int
    resident_bytes: int
    dirty_blocks_per_write: float
    snapshot_seconds: float
    t_cow_us_p50: float
    t_cow_us_p99: float
    fork_ms: float
    profile: PageProfile
    huge_pages: bool

    @property
    def peak_upper_bytes(self) -> int:
        return self.resident_bytes + self.n_blocks * self.block_bytes


def resolve(sc: Scenario, prof: PageProfile) -> Resolved:
    ps = prof.base_page
    huge = bool(sc.huge_pages and prof.huge_page)
    block_bytes = prof.huge_page if huge else ps
    pages_per_block = block_bytes // ps

    data_bytes = int(sc.data_gib * GIB)
    n_pages = int(np.ceil(data_bytes / (ps * sc.util)))
    n_blocks = int(np.ceil(n_pages / pages_per_block))

    # 一次逻辑写落在多少个复制单元上：大键值按字节占比折算均值
    mean_bytes = (sc.large_share * sc.large_mean_kib * 1024
                  + (1 - sc.large_share) * sc.small_mean_bytes)
    # 大键值天然是"物理上连续的多个块"，脏块数 ≈ ceil(均值/块大小)，小键值恒 1
    dirty = max(1.0, mean_bytes / block_bytes) * sc.write_amplification

    if sc.snapshot_seconds is not None:
        t_s = float(sc.snapshot_seconds)
    else:
        t_s = data_bytes / (max(1.0, sc.snapshot_mb_s) * MB)

    if huge:
        t50 = prof.thp_copy_us_p50 or 34.0
        t99 = prof.thp_copy_us_p99 or 78.0
    else:
        t50 = prof.cow_us_p50
        t99 = prof.cow_us_p99

    fork_ms = prof.fork_ms_base + prof.fork_ms_per_mpage * (n_pages / 1e6)
    return Resolved(ps, block_bytes, pages_per_block, n_pages, n_blocks,
                    data_bytes, n_pages * ps, dirty, t_s, t50, t99, fork_ms, prof, huge)


# ---------------------------------------------------------------------------
# 块热度
# ---------------------------------------------------------------------------
def distinct_touched(n_writes: float, n_blocks: int, sc: Scenario) -> float:
    """
    n 次写之后被写过的不同块数的期望 E(n)。

    这是整个模型的核心函数：同时决定内存峰值和 COW 停顿概率。

    用秩分组表示：组 g 有 s_g 个块，每块单次写命中概率 p_g。
    组内各块独立同分布时，组内期望命中数 = s_g (1 - (1-p_g)^n)，求和即可。
    分组对均匀/热键是精确的，对 Zipf 头部精确、尾部误差 < 1e-6。
    """
    from .heat import groups_for
    if n_writes <= 0 or n_blocks <= 0:
        return 0.0
    n = int(n_blocks)
    if sc.distribution == DIST_SEQUENTIAL:
        # 顺序写每次写前进一个"块"；大键值时一次写前进 d 个块
        return float(min(n_writes, n))

    sizes, probs = groups_for(sc, n)
    if sizes.size == 0:
        return 0.0
    return float(np.sum(sizes * (1.0 - np.power(np.maximum(1e-300, 1.0 - probs), n_writes))))


def new_copies_per_write(n_writes: float, n_blocks: int, sc: Scenario) -> float:
    """第 n 次写预期新增的复制块数。"""
    e1 = distinct_touched(n_writes, n_blocks, sc)
    e0 = distinct_touched(max(0.0, n_writes - 1.0), n_blocks, sc)
    return max(0.0, e1 - e0)


# ---------------------------------------------------------------------------
# 时间序列
# ---------------------------------------------------------------------------
@dataclass
class Series:
    t: np.ndarray
    copied_blocks: np.ndarray
    copied_bytes: np.ndarray
    memory_bytes: np.ndarray
    cow_stall_prob: np.ndarray     # 单次写命中未复制块的概率
    copy_rate: np.ndarray          # 每秒复制的块数
    cow_cpu_cores: np.ndarray
    stall_ms: np.ndarray           # 这一时刻命中 COW 的停顿时长（毫秒）
    pressure_stall_ms: np.ndarray  # 这一时刻内存压力导致的额外停顿（所有请求）
    used_fraction: np.ndarray      # 内存占用 / 物理内存
    fork_pause_ms: float = 0.0     # t=0 时 fork 复制页表造成的一次性停顿

    def peak(self) -> Dict[str, float]:
        i = int(np.argmax(self.memory_bytes))
        return {
            "peak_bytes": float(self.memory_bytes[i]),
            "peak_gib": float(self.memory_bytes[i] / GIB),
            "peak_time_s": float(self.t[i]),
            "copied_blocks": float(self.copied_blocks[i]),
            "copied_gib": float(self.copied_bytes[i] / GIB),
        }


def _pressure_stalls(used_gib: np.ndarray, sc: Scenario) -> np.ndarray:
    """
    内存压力造成的**全局**停顿（毫秒）。

    这是现场"延迟跳到几十毫秒"的主因，而不是 COW 拷贝本身（那只有几微秒）：
      * 占用 < high watermark（85%）：无额外停顿；
      * 85%~95%：内核开始直接回收 / khugepaged / 内存压缩，
        每次缺页都可能要等回收，按线性爬升到几十毫秒；
      * > 95%：直接回收同步等待，或触发 OOM killer（标成不可用）。
    经验参数：`reclaim_ms_at_100pct`（默认 35 ms）与 `reclaim_start`（0.85）。
    """
    ram = sc.ram_gib
    frac = used_gib / max(1e-9, ram)
    start = 0.85
    x = np.clip((frac - start) / (1.0 - start), 0.0, 1.0)
    # 凸增长：越接近满越陡
    return sc.reclaim_ms_at_100pct * np.power(x, 1.6)


def simulate(sc: Scenario, prof: PageProfile, n_steps: int = 180,
             pressure: bool = True, max_iter: int = 6) -> Tuple[Resolved, Series, Dict]:
    rv = resolve(sc, prof)
    t_base = rv.snapshot_seconds
    writes_per_s = sc.write_rate * rv.dirty_blocks_per_write

    stretch = 1.0
    series = None
    diag: Dict = {}
    for it in range(max_iter):
        T = t_base * stretch
        t = np.linspace(0.0, T, n_steps)
        n = writes_per_s * t
        e = np.array([distinct_touched(x, rv.n_blocks, sc) for x in n])
        de = np.array([new_copies_per_write(x, rv.n_blocks, sc) for x in n])
        # 单次逻辑写里至少有一个块是新的 → 停顿
        per_block_p = np.clip(de / max(1e-9, rv.dirty_blocks_per_write), 0, 1)
        stall = 1.0 - np.power(1.0 - per_block_p, rv.dirty_blocks_per_write)
        copy_rate = de * sc.write_rate
        cpu = copy_rate * rv.t_cow_us_p50 / 1e6
        # 物理上父进程最多再占一份数据区：复制字节不得超过 resident
        copied_bytes = np.minimum(e * rv.block_bytes, rv.resident_bytes)
        mem = rv.resident_bytes + copied_bytes
        mem_gib = mem / GIB
        used = mem_gib / max(1e-9, sc.ram_gib)
        pres = _pressure_stalls(mem_gib, sc)
        stall_ms = stall * rv.t_cow_us_p50 / 1000.0 + pres
        series = Series(t, e, copied_bytes, mem, stall, copy_rate, cpu,
                        stall_ms, pres, used, rv.fork_ms)

        peak_gib = float(mem_gib.max())
        rho = peak_gib / sc.ram_gib
        diag = {
            "rho": rho,
            "iterations": it + 1,
            "stretch": stretch,
            "peak_gib": peak_gib,
            "memory_ratio": float(mem.max() / max(1.0, rv.resident_bytes)),
            "copied_fraction": float(e[-1] / max(1, rv.n_blocks)),
            "writes_per_page": float(n[-1] / max(1, rv.n_pages)),
            "copy_cpu_peak": float(cpu.max()),
            "copy_cpu_mean": float(cpu.mean()),
            "seq_saturated": bool(cpu.max() >= max(1.0, sc.ncpu * 0.8)),
            "pressure_stall_peak_ms": float(pres.max()),
            "pressure_stall_mean_ms": float(pres.mean()),
            "used_fraction_peak": float(used.max()),
            # 内存超过 95% 且继续增长 → 现场就会看到 OOM killer
            "oom_risk": bool(peak_gib >= 0.97 * sc.ram_gib),
            "fork_ms": float(rv.fork_ms),
        }
        if not pressure:
            break
        if rho <= 0.85:
            new_stretch = 1.0
        else:
            new_stretch = 1.0 + 2.0 * min(1.0, (rho - 0.85) / 0.15)
        if abs(new_stretch - stretch) < 0.02:
            stretch = new_stretch
            break
        stretch = new_stretch
        if stretch >= 3.0:
            break

    diag["spiral"] = bool(diag["rho"] >= 1.0)
    return rv, series, diag


# ---------------------------------------------------------------------------
# 延迟
# ---------------------------------------------------------------------------
def latency(sc: Scenario, rv: Resolved, ser: Series) -> Dict[str, float]:
    """
    两部分相加：

      1. COW 拷贝停顿：一次写命中未复制块的概率 × 拷贝耗时（基页 ~1.3 µs，大页 ~45 µs）。
         这决定 p99/p99.9 会不会被"翻倍"级别的毫秒数抬高。
      2. 内存压力停顿：占用越过 85% 后全局生效，数十毫秒，是所有分位数一起变差的原因。

    分位判定：
      * p50  取平均 COW 命中概率 × 停顿；
      * p99  看重 COW 命中概率是否 > 1%，以及压力停顿；
      * p99.9 看是否 > 0.1%，并用偏保守的尾部系数。
    """
    stall = ser.cow_stall_prob
    pres = ser.pressure_stall_ms
    q_mean = float(stall.mean())
    q_max = float(stall.max())
    q_early = float(stall[: max(1, len(stall) // 10)].mean())
    t50 = rv.t_cow_us_p50 / 1000.0
    t99 = rv.t_cow_us_p99 / 1000.0
    cpu_peak = float(ser.cow_cpu_cores.max())
    headroom = max(1.0, sc.ncpu * 0.8)
    # COW 抢 CPU 造成的排队放大（拷贝本身之外的间接成本）
    cow_sat = 1.0 + max(0.0, (cpu_peak - headroom) / headroom) if cpu_peak > headroom else 1.0

    pres_peak = float(pres.max())
    pres_mean = float(pres.mean())

    cow_avg = q_mean * t50 * cow_sat
    cow_p99 = t99 * cow_sat if q_max > 0.01 else 0.0
    cow_p999 = t99 * 1.5 * cow_sat if q_max > 0.001 else 0.0

    # fork 的一次性停顿：单线程事件循环里，fork 期间所有请求都卡住。
    # 它只影响快照开始那一刻，但在 p99.9 / 最大值上必然出现。
    fork_ms = ser.fork_pause_ms
    p50 = sc.baseline_p50_ms + cow_avg + pres_mean
    p99 = sc.baseline_p99_ms + cow_p99 + pres_peak
    p999 = sc.baseline_p999_ms + cow_p999 + pres_peak * 1.6 + fork_ms
    return {
        "p50_ms": float(p50),
        "p99_ms": float(p99),
        "p999_ms": float(p999),
        "added_mean_ms": float(cow_avg + pres_mean),
        "added_p99_ms": float(p99 - sc.baseline_p99_ms),
        "added_p999_ms": float(p999 - sc.baseline_p999_ms),
        "cow_added_p99_ms": float(cow_p99),
        "pressure_added_p99_ms": float(pres_peak),
        "pressure_added_mean_ms": float(pres_mean),
        "stall_prob_start": q_max,
        "stall_prob_mean": q_mean,
        "stall_prob_early": q_early,
        "cow_cpu_cores_peak": cpu_peak,
        "cow_cpu_cores_mean": float(ser.cow_cpu_cores.mean()),
        "cow_saturation": float(cow_sat),
        "fork_pause_ms": float(fork_ms),
        "total_at_snapshot_start_ms": float(sc.baseline_p99_ms + fork_ms + t99),
    }


def evaluate(sc: Scenario, prof: PageProfile, pressure: bool = True,
             n_steps: int = 180) -> Dict:
    rv, ser, diag = simulate(sc, prof, pressure=pressure, n_steps=n_steps)
    lat = latency(sc, rv, ser)
    return {
        "scenario": sc.name,
        "data_gib": sc.data_gib,
        "util": sc.util,
        "write_rate": sc.write_rate,
        "distribution": sc.distribution,
        "distribution_label": DIST_LABELS.get(sc.distribution, sc.distribution),
        "locality": sc.locality,
        "huge_pages": rv.huge_pages,
        "page_size": rv.page_size,
        "block_kib": rv.block_bytes // 1024,
        "n_pages": rv.n_pages,
        "n_blocks": rv.n_blocks,
        "resident_gib": rv.resident_bytes / GIB,
        "snapshot_seconds": rv.snapshot_seconds,
        "writes_per_page": diag["writes_per_page"],
        "fork_ms": rv.fork_ms,
        "peak_gib": diag["peak_gib"],
        "peak_ratio": diag["memory_ratio"],
        "peak_upper_gib": rv.peak_upper_bytes / GIB,
        "copied_gib": float(ser.copied_bytes[-1] / GIB),
        "copied_fraction": diag["copied_fraction"],
        "rho": diag["rho"],
        "spiral": diag["spiral"],
        "seq_saturated": diag["seq_saturated"],
        "oom_risk": diag["oom_risk"],
        "pressure_stall_peak_ms": diag["pressure_stall_peak_ms"],
        "used_fraction_peak": diag["used_fraction_peak"],
        "latency": lat,
    }
