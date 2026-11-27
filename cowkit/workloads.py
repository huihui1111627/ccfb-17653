"""四种真实业务写入模式。

每个 workload 是一个生成器，按 *每秒写入数* 吐出 (t_seconds, op) 事件；
op 为字典，描述这次写要弄脏多少页、是不是热键，engine 据此驱动页模型。

模式
----
uniform     : 均匀随机写（典型缓存，键空间大、访问均匀）
hotkey      : 少数热键反复写（5% 的键吃掉 80% 写入）
sequential  : 批量导入顺序写（整段连续页被扫过，COW 几乎线性增长）
mixed_sizes : 大键值 + 海量小键混合（大键一次写脏很多页）
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class WorkloadSpec:
    name: str
    write_rate: float          # 每秒写请求数
    duration: float            # 秒（通常等于快照耗时）
    seed: int = 0

    @property
    def n_writes(self) -> int:
        return int(round(self.write_rate * self.duration))


def _times(rng: np.random.Generator, rate: float, duration: float) -> np.ndarray:
    """泊松到达：请求间隔服从指数分布，返回相对快照开始的秒级时间戳。"""
    n = int(rate * duration)
    intervals = rng.exponential(1.0 / max(rate, 1e-9), size=n)
    t = np.cumsum(intervals)
    return t[t < duration]


def uniform_writes(spec: WorkloadSpec):
    """均匀随机写，每次 1 个页（小键值）。"""
    rng = np.random.default_rng(spec.seed)
    for t in _times(rng, spec.write_rate, spec.duration):
        yield float(t), {"kind": "uniform", "pages": 1}


def hotkey_writes(spec: WorkloadSpec, hot_frac: float = 0.05, hot_share: float = 0.8):
    """热键写：hot_frac 的键占 hot_share 流量。"""
    rng = np.random.default_rng(spec.seed + 1)
    for t in _times(rng, spec.write_rate, spec.duration):
        hot = rng.random() < hot_share
        yield float(t), {"kind": "hot" if hot else "cold", "pages": 1,
                         "hot_frac": hot_frac}


def sequential_writes(spec: WorkloadSpec, pages_per_write: int = 1):
    """批量导入顺序写：访问位置随时间线性推进。"""
    rng = np.random.default_rng(spec.seed + 2)
    cursor = 0
    for t in _times(rng, spec.write_rate, spec.duration):
        yield float(t), {"kind": "sequential", "pages": pages_per_write,
                         "cursor": cursor}
        cursor += pages_per_write


def mixed_size_writes(spec: WorkloadSpec, large_ratio: float = 0.08,
                      large_pages_mean: float = 16.0):
    """大键+小键混合：8% 是大键（几何分布跨页，均值 16 页 ≈ 64KiB@4K），其余小键 1 页。"""
    rng = np.random.default_rng(spec.seed + 3)
    for t in _times(rng, spec.write_rate, spec.duration):
        if rng.random() < large_ratio:
            span = max(2, int(rng.geometric(1.0 / large_pages_mean)))
            yield float(t), {"kind": "large", "pages": span}
        else:
            yield float(t), {"kind": "small", "pages": 1}


WORKLOADS = {
    "uniform": uniform_writes,
    "hotkey": hotkey_writes,
    "sequential": sequential_writes,
    "mixed": mixed_size_writes,
}


def expected_dirty_fraction(workload: str, writes: int, n_mapped_pages: int,
                            hot_frac: float = 0.05, hot_share: float = 0.8,
                            avg_pages_per_write: float = 1.0) -> float:
    """闭式预估：writes 次写之后，被触碰过的已映射页比例（基础页、未开大页）。

    uniform/mixed : 优惠券收集模型 1-(1-1/N)^(k)，k=总页触碰次数
    hotkey        : 热区几乎必脏 + 冷区按其流量做收集
    sequential    : min(1, 总触碰页数/N)
    """
    n = max(1, n_mapped_pages)
    k = writes * avg_pages_per_write
    if workload == "sequential":
        return min(1.0, k / n)
    if workload == "hotkey":
        n_hot = max(1, n * hot_frac)
        n_cold = n - n_hot
        k_hot = writes * hot_share
        k_cold = writes * (1 - hot_share)
        f_hot = 1.0 - math.exp(-k_hot / n_hot)
        f_cold = (1.0 - math.exp(-k_cold / max(1, n_cold))) if n_cold > 0 else 0.0
        return (f_hot * n_hot + f_cold * n_cold) / n
    # uniform / mixed 同属独立随机选页
    return 1.0 - math.exp(-k / n)
