"""
业务写入模式：把"每秒多少次写、写到哪里"变成**页/块号轨迹**。

四种形态（与 heat.py 共用同一套秩分组分布，保证解析模型和逐页仿真
看到的是同一个热度曲线）：

  均匀随机写       每块等概率。
  少数热键反复写    热度高度集中，E(n) 很快饱和 → 峰值几乎不涨；
                  但它把快照头几秒的每次写都变成 COW，延迟最先恶化。
  Zipf 长尾        头部集中、尾部很长，快照后期仍在触碰新页。
  顺序写/批量导入   按块号严格推进，E(n)=min(n,块数)。

实现要点：
  * 用逆 CDF 在"秩分组"上抽样：先抽组（按组质量），再在组内均匀取秩。
    这样与 heat.py 的 (s_g, p_g) 表示严格一致。
  * 顺序写维护 cursor，按块号递增，大键值一次前进 d 个块。
  * 大键值每次写产生连续 d 个块，跨越块边界时自动拼接，去重。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .heat import groups_for
from .model import DIST_SEQUENTIAL, Scenario, resolve
from .profiles import get as get_profile


class BlockPattern:
    """在**复制单元**（基页或大页块）上采样的写入轨迹。"""

    def __init__(self, sc: Scenario, n_blocks: int, seed: int = 0):
        self.sc = sc
        self.n = int(n_blocks)
        self.rng = np.random.default_rng(seed)
        self.sequential = sc.distribution == DIST_SEQUENTIAL
        self.cursor = 0
        rv = resolve(sc, get_profile(sc.profile))
        self.d = max(1, int(round(rv.dirty_blocks_per_write)))

        sizes, probs = groups_for(sc, self.n)
        self.g_sizes = sizes.astype(np.int64)
        # groups_for 给的是"每块命中概率"，组质量 = 块数 × 每块概率
        mass = sizes * probs
        self.g_mass = mass / mass.sum()
        self.g_cdf = np.cumsum(self.g_mass)
        self.g_start = np.concatenate(([0], np.cumsum(self.g_sizes)[:-1])).astype(np.int64)

    # ------------------------------------------------------------------
    def draw(self, m: int) -> np.ndarray:
        """抽 m 次逻辑写，返回压平后的块号（含大键值的连续多块）。"""
        if self.sequential:
            total = m * self.d
            idx = (self.cursor + np.arange(total)) % self.n
            self.cursor = (self.cursor + total) % self.n
            return idx

        g = np.searchsorted(self.g_cdf, self.rng.random(m), side="right")
        g = np.minimum(g, self.g_sizes.size - 1)
        offset = (self.rng.random(m) * self.g_sizes[g]).astype(np.int64)
        base = self.g_start[g] + offset
        if self.d <= 1:
            return base
        steps = np.arange(self.d, dtype=np.int64)
        return ((base[:, None] + steps[None, :]) % self.n).reshape(-1)


# 兼容旧名字
WritePattern = BlockPattern
