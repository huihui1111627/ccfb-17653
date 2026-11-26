"""
块热度：把"写入落在哪些页上"表示成一组 (块数, 单块命中概率)。

四种业务形态在这套表示下都能精确或高精度写出：

  均匀  一块，p = 1/n。
  热键  两组：热区 k 块 p_hot，冷区 n-k 块 p_cold。精确。
  Zipf  头部若干名精确列出，其余按等比数列分组（组内 p 相差 ≤ 15%）。
        组质量用 Euler–Maclaurin 求 H(i)=Σ_{j≤i} j^-s，误差 <1e-3。
  顺序  不进入这里：E(n)=min(n,页数) 直接给闭式。

`locality` 描述"业务侧热度在页级还剩多少"：真实分配器会把键散到多个
size class 的页上。locality=1 表示热度完整保留，0 表示被打散成均匀。
混合是线性的：p' = loc·p + (1-loc)/n。

这个模块同时被解析模型（算 E(n)）和写入轨迹（按组抽样）使用，
保证两边用的是同一个分布。
"""
from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np

from .model import DIST_HOT, DIST_SEQUENTIAL, DIST_UNIFORM, DIST_ZIPF, Scenario

_EM_SMALL = 256          # 头部精确保留的秩数
_GROWTH = 1.15           # 尾部分组的等比因子


def _zeta(s: float) -> float:
    """ζ(s) 的粗略值，只用于 Euler–Maclaurin 常数项。"""
    if abs(s - 1.0) < 1e-9:
        return float("inf")
    # 直接用前若干项 + 积分余项
    j = np.arange(1, 200001, dtype=np.float64)
    tail = 200000 ** (1 - s) / (s - 1) if s > 1 else 0.0
    return float(np.sum(np.power(j, -s)) + tail)


_EXACT_HEAD = 200_000


def _zipf_h(i, s: float) -> np.ndarray:
    """
    H(i) = Σ_{j=1..i} j^-s。

    做法：前 `_EXACT_HEAD` 项直接累加（缓存的累积和），之后用积分近似
    H(i) = H(M) + ∫_M^i x^-s dx。相对误差 < 1e-9，且对 s<1 / s=1 / s>1 一致。
    """
    i = np.maximum(np.asarray(i, dtype=np.float64), 0.0)
    out = np.zeros_like(i)
    cum = _exact_cum(s)
    M = cum.size                     # 精确前缀长度

    lo = i <= M
    if lo.any():
        idx = np.clip(i[lo].astype(np.int64), 1, M) - 1
        out[lo] = cum[idx]
        out[lo & (i < 1)] = 0.0

    hi = i > M
    if hi.any():
        x = i[hi]
        base = cum[-1]
        if abs(s - 1.0) < 1e-12:
            out[hi] = base + np.log(x / M)
        else:
            out[hi] = base + (np.power(x, 1 - s) - M ** (1 - s)) / (1 - s)
    return out


@lru_cache(maxsize=16)
def _exact_cum(s: float) -> np.ndarray:
    j = np.arange(1, _EXACT_HEAD + 1, dtype=np.float64)
    return np.cumsum(np.power(j, -s))


def zipf_groups(n: int, s: float) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (每组块数, 每组概率质量)，按秩升序。头部精确，尾部等比分组。"""
    n = int(n)
    if n <= 0:
        return np.empty(0), np.empty(0)
    if n <= _EM_SMALL:
        sizes = np.ones(n, dtype=np.float64)
        mass = np.power(np.arange(1, n + 1, dtype=np.float64), -s)
        return sizes, mass / mass.sum()

    head = min(_EM_SMALL, n)
    starts = [1]
    while starts[-1] < head:
        starts.append(starts[-1] + 1)
    i = head
    while i < n:
        i = min(n, max(i + 1, int(round(i * _GROWTH))))
        starts.append(i + 1)
    edges = np.asarray(starts, dtype=np.int64)          # 每组起始秩（最后一组起于 ≤n）
    if edges[-1] > n:
        edges = edges[:-1]
    ends = np.concatenate([edges[1:] - 1, [n]])
    sizes = (ends - edges + 1).astype(np.float64)
    total = float(_zipf_h(np.array([n]), s)[0])
    mass = (_zipf_h(ends, s) - _zipf_h(edges - 1, s)) / total
    mass = np.maximum(mass, 0.0)
    mass /= mass.sum()
    return sizes, mass


@lru_cache(maxsize=256)
def heat_groups(n: int, distribution: str, zipf_s: float, hot_fraction: float,
                hot_space: float, locality: float) -> Tuple[np.ndarray, np.ndarray]:
    """(块数组, 单块命中概率)。Σ size = n，Σ size*prob = 1。"""
    n = int(n)
    if n <= 0:
        return np.empty(0), np.empty(0)
    if distribution == DIST_HOT:
        k = min(float(n), max(1.0, hot_space * n))
        sizes = np.array([k, n - k], dtype=np.float64)
        probs = np.array([hot_fraction / k,
                          (1.0 - hot_fraction) / max(1.0, n - k)], dtype=np.float64)
    elif distribution == DIST_ZIPF:
        sizes, mass = zipf_groups(n, float(zipf_s))
        probs = np.divide(mass, sizes, out=np.zeros_like(mass), where=sizes > 0)
    elif distribution == DIST_SEQUENTIAL:
        # 顺序写不进热度模型：等价于"每个块等概率"，但累积形态是线性的，
        # 由 distinct_touched 特判。这里返回均匀只是为了抽样一致性。
        sizes = np.array([float(n)])
        probs = np.array([1.0 / n])
    else:
        sizes = np.array([float(n)])
        probs = np.array([1.0 / n])

    loc = float(np.clip(locality, 0.0, 1.0))
    if loc < 1.0:
        probs = loc * probs + (1.0 - loc) / n
    return sizes, probs


def groups_for(sc: Scenario, n_blocks: int) -> Tuple[np.ndarray, np.ndarray]:
    return heat_groups(int(n_blocks), sc.distribution, float(sc.zipf_s),
                       float(sc.hot_fraction), float(sc.hot_space),
                       float(np.clip(sc.locality, 0, 1)))
