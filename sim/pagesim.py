"""
页级 Monte Carlo：不复用解析公式，直接按位图/桶推演 fork 之后的复制过程。

解析模型的 `E(n) = Σ(1-(1-w_i)^n)` 是**独立同分布期望**。真实写入有它给不出的东西：

  1. 落点离散，同样 n 次写的实际复制数有方差；
  2. 热键/局部性下"同一页两次写"的间隔远小于均匀假设；
  3. 大键值会连续跨越块边界，实际触及的块数比按均值折算的多一点；
  4. 快照期间内存涨到 85% 以上时，全局回收停顿会和 COW 事件叠加。

所以这里给两套独立实现，和解析公式互相印证：

  * `BitSetSim` 逐页位图，页数 ≤ 300 万时最精确，用于小规模与参数校验；
  * `BucketSim` 把页空间压到 4096 个桶、每桶维护"已复制比例"，
                可推到 800 万页并保持统计一致，用于 30 GiB 场景。

页面上的主曲线用这两个仿真的输出；解析公式只用于瞬时参数扫描。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .model import DIST_SEQUENTIAL, Scenario, resolve
from .profiles import get as get_profile
from .workload import BlockPattern as WritePattern


# ---------------------------------------------------------------------------
# 输出结构
# ---------------------------------------------------------------------------
@dataclass
class SimTrace:
    t: np.ndarray                    # 秒
    copied_blocks: np.ndarray        # 累计复制块数
    memory_bytes: np.ndarray         # 父进程数据区物理占用
    resident_bytes: int
    block_bytes: int
    writes: np.ndarray               # 累计逻辑写次数
    n_blocks: int
    peak_gib: float = 0.0
    # 延迟采样（与 COW 事件对齐）
    lat_t: np.ndarray = field(default_factory=lambda: np.empty(0))
    lat_ms: np.ndarray = field(default_factory=lambda: np.empty(0))
    lat_cow: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    lat_pressure: np.ndarray = field(default_factory=lambda: np.empty(0))
    # 热度
    heat: Optional[np.ndarray] = None
    bucket_span: float = 1.0

    def summary(self) -> Dict[str, float]:
        i = int(np.argmax(self.memory_bytes))
        lat = self.lat_ms
        cow = self.lat_cow
        return {
            "peak_gib": float(self.memory_bytes[i] / (1 << 30)),
            "peak_time_s": float(self.t[i]),
            "peak_ratio": float(self.memory_bytes[i] / max(1, self.resident_bytes)),
            "copied_blocks": int(self.copied_blocks[-1]),
            "copied_bytes": int(self.copied_blocks[-1] * self.block_bytes),
            "copied_fraction": float(self.copied_blocks[-1] / max(1, self.n_blocks)),
            "duration_s": float(self.t[-1]),
            "writes": int(self.writes[-1]),
            "lat_p50_ms": float(np.percentile(lat, 50)) if lat.size else 0.0,
            "lat_p99_ms": float(np.percentile(lat, 99)) if lat.size else 0.0,
            "lat_p999_ms": float(np.percentile(lat, 99.9)) if lat.size else 0.0,
            "cow_hit_rate": float(cow.mean()) if cow.size else 0.0,
            "cow_hit_rate_early": float(cow[: max(1, cow.size // 10)].mean()) if cow.size else 0.0,
        }


def _pressure_ms(mem_bytes: float, sc: Scenario) -> float:
    """内存压力导致的全局停顿（毫秒），与解析模型同一口径。"""
    frac = mem_bytes / (sc.ram_gib * (1 << 30))
    x = min(1.0, max(0.0, (frac - 0.85) / 0.15))
    return sc.reclaim_ms_at_100pct * x ** 1.6


def sample_latency(pages_iter, n_blocks: int, block_pages: int, rv, sc: Scenario,
                   total_writes: int, seed: int, n_take: int = 3000) -> Tuple[np.ndarray, ...]:
    """
    给定一个能按批产出页号的生成器，重放一遍写入序列，在均匀间隔的写序号上
    记录"是否命中尚未复制的块"以及该请求的延迟。
    """
    seen = np.zeros(n_blocks, dtype=bool)
    if total_writes <= 0:
        e = np.empty(0)
        return e, e, np.empty(0, dtype=bool), e

    n_take = int(min(n_take, total_writes))
    picks = np.unique(np.linspace(0, total_writes - 1, n_take).astype(np.int64))
    rng = np.random.default_rng(seed + 999)
    resident = rv.n_pages * rv.page_size
    block_bytes = rv.block_bytes

    out_t, out_ms, out_cow, out_pres = [], [], [], []
    pos = 0
    n_copied = 0
    for target in picks:
        m = int(target - pos)
        if m < 0:
            m = 0
        pos = int(target)
        if m > 0:
            batch = pages_iter(m)
            blocks = np.minimum(batch // block_pages, n_blocks - 1)
            fresh = np.unique(blocks)
            n_copied += int((~seen[fresh]).sum())
            seen[fresh] = True
        one = pages_iter(1)
        blk = int(min(one[0] // block_pages, n_blocks - 1))
        is_cow = not seen[blk]
        if is_cow:
            seen[blk] = True
            n_copied += 1
        mem = resident + n_copied * block_bytes
        pres = _pressure_ms(mem, sc)
        if is_cow:
            cow_ms = (rv.t_cow_us_p99 if rng.random() > 0.9 else rv.t_cow_us_p50) / 1000.0
        else:
            cow_ms = 0.0
        out_t.append(target / max(1e-9, sc.write_rate))
        out_ms.append(sc.baseline_p50_ms + cow_ms + pres)
        out_cow.append(is_cow)
        out_pres.append(pres)
    return (np.asarray(out_t), np.asarray(out_ms),
            np.asarray(out_cow, dtype=bool), np.asarray(out_pres))


# ---------------------------------------------------------------------------
# 精确位图
# ---------------------------------------------------------------------------
class BitSetSim:
    """逐页位图，页数 × 1 字节内存。≤ 400 万块时最准确。"""

    def __init__(self, sc: Scenario, seed: int = 0, n_samples: int = 240,
                 n_pages: Optional[int] = None):
        self.sc = sc
        self.prof = get_profile(sc.profile)
        self.rv = resolve(sc, self.prof)
        if n_pages:
            self.rv.n_pages = int(n_pages)
            self.rv.n_blocks = int(np.ceil(n_pages / self.rv.pages_per_block))
        self.seed = seed
        self.n_samples = n_samples
        self.block_pages = self.rv.pages_per_block
        self.n_blocks = self.rv.n_blocks

    def _page_iter(self, pattern: WritePattern):
        buf = np.empty(0, dtype=np.int64)

        def it(m: int) -> np.ndarray:
            nonlocal buf
            if m <= 0:
                return np.empty(0, dtype=np.int64)
            return pattern.draw(m)
        return it

    def run(self) -> SimTrace:
        rv, sc = self.rv, self.sc
        nb = self.n_blocks
        bp = self.block_pages
        resident = rv.n_pages * rv.page_size
        block_bytes = rv.block_bytes
        total_writes = int(round(sc.write_rate * rv.snapshot_seconds))
        pattern = WritePattern(sc, nb, seed=self.seed)

        copied = np.zeros(nb, dtype=bool)
        pts = np.unique(np.linspace(0, total_writes, self.n_samples + 1)[1:].astype(np.int64))
        t_out = np.empty(pts.size)
        c_out = np.empty(pts.size)
        m_out = np.empty(pts.size)
        w_out = np.empty(pts.size)

        pos = 0
        for i, target in enumerate(pts):
            m = int(target - pos)
            pos = int(target)
            if m > 0:
                blocks = pattern.draw(m)
                copied[np.minimum(blocks, nb - 1)] = True
            t_out[i] = pos / sc.write_rate
            c_out[i] = copied.sum()
            m_out[i] = resident + min(c_out[i] * block_bytes, resident)
            w_out[i] = pos

        # 延迟采样：独立 seed 重放
        pat2 = WritePattern(sc, nb, seed=self.seed + 1)
        lt, lm, lc, lp = sample_latency(pat2.draw, nb, 1, rv, sc,
                                        total_writes, self.seed)
        tr = SimTrace(t=t_out, copied_blocks=c_out, memory_bytes=m_out,
                      resident_bytes=resident, block_bytes=block_bytes,
                      writes=w_out, n_blocks=nb,
                      lat_t=lt, lat_ms=lm, lat_cow=lc, lat_pressure=lp,
                      heat=self._heat())
        tr.peak_gib = float(m_out.max() / (1 << 30))
        return tr

    def _heat(self):
        from .heat import groups_for
        sizes, probs = groups_for(self.sc, self.rv.n_blocks)
        return (sizes, sizes * probs)


# ---------------------------------------------------------------------------
# 桶近似（大规模）
# ---------------------------------------------------------------------------
class BucketSim:
    """
    把 n_blocks 个块分到 B 个桶。每个桶维护被复制块数 f_b ∈ [0, g_b]。

    单次写落在桶 b 的某一块上：若该块未复制则 f_b += 1。桶内均匀随机假设下
    期望增量 = (g_b - f_b)/g_b。用这个确定性递推代替逐块位图，
    误差来自桶内不均匀（每桶只有 g 块，g ≤ 数百，误差 <1%）。
    """

    def __init__(self, sc: Scenario, seed: int = 0, n_samples: int = 240,
                 n_buckets: int = 4096):
        self.sc = sc
        self.prof = get_profile(sc.profile)
        self.rv = resolve(sc, self.prof)
        self.seed = seed
        self.n_samples = n_samples
        self.n_buckets = n_buckets

    def run(self) -> SimTrace:
        rv, sc = self.rv, self.sc
        nb = rv.n_blocks
        bp = rv.pages_per_block
        B = min(self.n_buckets, nb)
        g = nb / B
        resident = rv.n_pages * rv.page_size
        block_bytes = rv.block_bytes
        total_writes = int(round(sc.write_rate * rv.snapshot_seconds))

        pattern = WritePattern(sc, nb, seed=self.seed)
        # 桶级写入概率 = 组质量按块区间落到桶上
        from .heat import groups_for
        sizes, probs = groups_for(sc, nb)
        mass = sizes * probs
        mass = mass / mass.sum()
        g_start = np.concatenate(([0], np.cumsum(sizes)[:-1]))
        bp_w = np.zeros(B)
        for sz, mm, st in zip(sizes, mass, g_start):
            lo = int(st); hi = int(min(nb, st + sz))
            if hi <= lo:
                continue
            per_block = mm / (hi - lo)
            b_lo = min(B - 1, int(lo / g))
            b_hi = min(B - 1, int((hi - 1) / g))
            for b in range(b_lo, b_hi + 1):
                a = max(lo, b * g); z = min(hi, (b + 1) * g)
                if z > a:
                    bp_w[b] += per_block * (z - a)
        bp_w = np.maximum(bp_w, 0.0)
        bp_w /= bp_w.sum()

        copied = np.zeros(B)
        seq_cursor = 0.0
        pts = np.unique(np.linspace(0, total_writes, self.n_samples + 1)[1:].astype(np.int64))
        t_out = np.empty(pts.size)
        c_out = np.empty(pts.size)
        m_out = np.empty(pts.size)
        w_out = np.empty(pts.size)

        pos = 0
        for i, target in enumerate(pts):
            m = int(target - pos)
            pos = int(target)
            if m > 0:
                if sc.distribution == DIST_SEQUENTIAL:
                    # 顺序写：本批 m 次写覆盖 m*d 个连续块
                    span = m * max(1, round(rv.dirty_blocks_per_write))
                    start = seq_cursor
                    end = min(nb, start + span)
                    for b in range(int(start / g), min(B, int((end - 1) / g) + 1)):
                        a = max(start, b * g); z = min(end, (b + 1) * g)
                        copied[b] += max(0, z - a)
                    seq_cursor = min(nb, start + span)
                else:
                    # 每个桶这一批被写到的期望次数
                    hits = m * bp_w
                    room = np.maximum(0.0, g - copied)
                    # 桶内均匀：期望新增复制 = g - (g-copied)*exp(-hits/g)
                    new = g - (g - copied) * np.exp(-hits / g)
                    copied = np.minimum(g, new)
                copied = np.minimum(g, copied)
            t_out[i] = pos / sc.write_rate
            c_out[i] = copied.sum()
            m_out[i] = resident + min(c_out[i] * block_bytes, resident)
            w_out[i] = pos

        pat2 = WritePattern(sc, nb, seed=self.seed + 1)
        lt, lm, lc, lp = sample_latency(pat2.draw, nb, 1, rv, sc,
                                        total_writes, self.seed)
        tr = SimTrace(t=t_out, copied_blocks=c_out, memory_bytes=m_out,
                      resident_bytes=resident, block_bytes=block_bytes,
                      writes=w_out, n_blocks=nb,
                      lat_t=lt, lat_ms=lm, lat_cow=lc, lat_pressure=lp,
                      heat=self._heat())
        tr.peak_gib = float(m_out.max() / (1 << 30))
        return tr

    def _heat(self):
        from .heat import groups_for
        sizes, probs = groups_for(self.sc, self.rv.n_blocks)
        return (sizes, sizes * probs)


def run_sim(sc: Scenario, seed: int = 0, n_samples: int = 240,
            force_buckets: bool = False) -> SimTrace:
    """按规模自动选实现：≤ 400 万块走位图，否则走桶近似。"""
    rv = resolve(sc, get_profile(sc.profile))
    if force_buckets or rv.n_blocks > 4_000_000:
        return BucketSim(sc, seed=seed, n_samples=n_samples).run()
    return BitSetSim(sc, seed=seed, n_samples=n_samples).run()
