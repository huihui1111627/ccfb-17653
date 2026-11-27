"""COW 被复制页数的闭式公式 —— 用来和逐页仿真互相校验。

记号
----
N   : 已映射基础页数（活页+空洞页；空洞页被写一样复制）
w(t): 快照开始到 t 时刻的累计写请求数
q   : 每次写平均触碰的基础页数（小键=1，大键按均值）
H   : 大页大小 / 基础页大小（4K 页 + 2M THP => H=512）；不开大页 H=1
f   : 已映射页在大页内的占满度（接近 1，碎片越多略小）

基础页（THP 关）
----------------
均匀/混合随机写（优惠券收集）:
    C(t) = N * (1 - (1 - 1/N)^(w*q))  ≈ N*(1-e^(-wq/N))
热键写（热区 h 比例占 s 流量）:
    热区 N_h=hN 几乎全脏，冷区 N_c=(1-h)N 按 (1-s)w 的流量收集
顺序写:
    C(t) = min(N, w*q)   —— 线性，直到扫完

透明大页（THP 开）
------------------
随机写触碰的是“大页”这个容器，复制单位变成大页：
    C_H(t) = f * H * N_H * (1 - e^(-w*q / N_H))，其中 N_H = N/H（向上取整）
顺序写: C_H(t) = f * min(N, H*ceil(w*q/H))
热键:   热区涉及的大页很快全部复制，冷区大页按冷流量收集。

峰值内存
--------
parent_peak = 基础常驻 R0
            + fork 后被复制的页字节 C_peak * 页大小
            + 快照期间新分配（新键/变大键）的字节
（子进程不额外占物理内存，它只持有指向共享帧/父进程私有副本的页表。）
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class AnalyticInput:
    n_mapped_pages: int
    page_size: int
    writes_total: int
    avg_pages_per_write: float = 1.0
    workload: str = "uniform"
    huge_size: int = 0
    huge_fill: float = 0.95
    hot_frac: float = 0.05
    hot_share: float = 0.8
    new_alloc_bytes: int = 0


def _huge_factor(cfg: AnalyticInput) -> tuple[int, int]:
    if cfg.huge_size <= 0:
        return 1, 1
    h = max(1, cfg.huge_size // cfg.page_size)
    n_huge = math.ceil(cfg.n_mapped_pages / h)
    return h, n_huge


def copied_pages_curve(cfg: AnalyticInput, w_series) -> list[float]:
    """给定累计写次数序列 w_series，返回被复制基础页数序列（期望值）。"""
    n = max(1, cfg.n_mapped_pages)
    h, n_huge = _huge_factor(cfg)
    q = cfg.avg_pages_per_write
    out = []
    for w in w_series:
        if h == 1:
            out.append(_base_copied(cfg, n, w, q))
        else:
            out.append(_huge_copied(cfg, n, n_huge, h, w, q))
    return out


def _base_copied(cfg: AnalyticInput, n: int, w: float, q: float) -> float:
    k = w * q
    if cfg.workload == "sequential":
        return min(float(n), k)
    if cfg.workload == "hotkey":
        n_hot = max(1.0, n * cfg.hot_frac)
        n_cold = max(0.0, n - n_hot)
        f_hot = 1.0 - math.exp(-w * cfg.hot_share / n_hot)
        f_cold = 1.0 - math.exp(-w * (1 - cfg.hot_share) / max(1.0, n_cold))
        return f_hot * n_hot + f_cold * n_cold
    return n * (1.0 - math.exp(-k / n))


def _huge_copied(cfg: AnalyticInput, n: int, n_huge: int, h: int,
                 w: float, q: float) -> float:
    # 大页视角：每次写落到某个大页；复制后按大页内已映射占满度折算基础页
    f = cfg.huge_fill
    if cfg.workload == "sequential":
        touched_huge = min(n_huge, math.ceil(w * q / h))
        return min(float(n), touched_huge * h * f)
    if cfg.workload == "hotkey":
        nh_huge = max(1, n_huge * cfg.hot_frac)
        nc_huge = max(1, n_huge - nh_huge)
        f_hot = 1.0 - math.exp(-w * cfg.hot_share / nh_huge)
        f_cold = 1.0 - math.exp(-w * (1 - cfg.hot_share) / nc_huge)
        touched_huge = f_hot * nh_huge + f_cold * nc_huge
        return min(float(n), touched_huge * h * f)
    touched_huge = n_huge * (1.0 - math.exp(-w * q / n_huge))
    return min(float(n), touched_huge * h * f)


def peak_memory(cfg: AnalyticInput, rss_base_bytes: int) -> dict:
    """返回峰值内存分解（字节）。"""
    peak_copied = copied_pages_curve(cfg, [cfg.writes_total])[0]
    copied_bytes = peak_copied * cfg.page_size
    peak = rss_base_bytes + copied_bytes + cfg.new_alloc_bytes
    return {
        "rss_base_bytes": rss_base_bytes,
        "copied_pages_peak": peak_copied,
        "copied_bytes": copied_bytes,
        "new_alloc_bytes": cfg.new_alloc_bytes,
        "peak_bytes": peak,
        "copy_overhead_pct": 100.0 * copied_bytes / max(1, rss_base_bytes),
    }


def time_to_copy_fraction(cfg: AnalyticInput, target_frac: float,
                          write_rate: float) -> float:
    """反解：复制页数达到总量 target_frac 比例时，距快照开始多少秒（随机写）。

    容器视角（基础页容器=页本身，THP 容器=大页），收集曲线 1-e^(-wq/C)=p，
    故 w = -C·ln(1-p)/q，再除以写速率得秒。
    """
    n = max(1, cfg.n_mapped_pages)
    h, n_huge = _huge_factor(cfg)
    containers = n_huge if h > 1 else n
    w = -containers * math.log(max(1e-12, 1.0 - target_frac)) \
        / max(1e-9, cfg.avg_pages_per_write)
    return w / max(write_rate, 1e-9)
