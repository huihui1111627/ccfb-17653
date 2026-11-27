"""键值数据在内存页上的分布模型。

真实规律（Linux / POSIX）
------------------------
1. 进程拿到的匿名内存按 *基础页* （Linux x86_64 = 4 KiB，本机 macOS arm64 = 16 KiB）
   映射。分配器（jemalloc 等）把对象塞进页内；小键值共享同一个页，大键值跨多个页。
2. 删除/过期键只是把字节范围还给分配器，物理页通常不会立刻 munmap，于是留下页内空洞，
   页仍处于已映射状态 —— fork 之后它照样是一个可被写脏、可被复制的页。
3. fork() 后父子共享全部页，内核把页表项降为只读。父进程第一次写某个页 ->
   缺页异常 -> 内核复制整页 -> 父进程拿到自己的可写副本，这就是 Copy-On-Write。
4. 透明大页（THP，Linux 2 MiB/页）开启时，复制单位从基础页变成整个大页：
   写大页内任意一个字节，整个 2 MiB 都要复制。

这里提供两个同接口的模型：
- KeySpace  : 逐键模拟（小场景/和真实 fork 实测对齐用），精确跟踪每个键的字节范围；
- PageArena : 逐页统计模型（30 GiB 级大场景用），按相同的占用/空洞统计规律直接建页。
两者输出的“写一批键弄脏哪些页”在统计意义上一致，sweep 大场景只能用 PageArena。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Linux x86_64 透明大页（THP）标准大小 = 2 MiB（不是 2 GiB！）
THP_2M = 2 * 1024 * 1024
# 基础页大小
PAGE_4K = 4096

# 页状态位
PAGE_FREE = 0       # 未映射（没有数据，fork 后也不会被复制）
PAGE_HOLE = 1       # 已映射但全是空洞（删剩的空页，仍可被写脏）
PAGE_LIVE = 2       # 含有活数据


@dataclass
class ArenaConfig:
    """内存竞技场配置。

    data_bytes      : 活数据逻辑大小（键+值的有效字节）
    page_size       : 基础页大小（字节）
    huge_size       : 大页大小（字节），0 表示不开大页
    fill_fraction   : 活数据占已映射内存的比例（其余为分配器碎片+空洞），
                      jemalloc 下典型 0.75~0.95；越小空洞越多
    alloc_overhead  : 分配器元数据/外部碎片相对活数据的额外占用系数
    """

    data_bytes: int
    page_size: int = 4096
    huge_size: int = 0
    fill_fraction: float = 0.85
    alloc_overhead: float = 0.08
    seed: int = 0

    @property
    def mapped_bytes(self) -> int:
        """活数据 + 分配器开销折算成的已映射字节（空洞页也含在里面）。"""
        return int(self.data_bytes * (1.0 + self.alloc_overhead) / self.fill_fraction)

    @property
    def n_pages(self) -> int:
        return math.ceil(self.mapped_bytes / self.page_size)

    @property
    def n_huge(self) -> int:
        if self.huge_size <= 0:
            return 0
        return math.ceil(self.mapped_bytes / self.huge_size)


def pages_for_span(start: int, length: int, page_size: int) -> np.ndarray:
    """字节区间 [start, start+length) 覆盖的全部基础页号（含首尾的部分页）。"""
    if length <= 0:
        return np.empty(0, dtype=np.int64)
    first = start // page_size
    last = (start + length - 1) // page_size
    return np.arange(first, last + 1, dtype=np.int64)


class PageArena:
    """逐页统计模型：O(页数) 内存，可以直接建 30 GiB 级场景。

    内部只存一个长度 = 基础页数的状态数组（PAGE_*），不存具体键。
    写操作以“弄脏的页集合”为单位驱动 COW 引擎，键大小分布的影响通过
    workloads 模块里的跨页规则体现（小键=1页、大键=连续多页、顺序写=连续区间）。
    """

    def __init__(self, cfg: ArenaConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        n = cfg.n_pages
        self.state = np.zeros(n, dtype=np.uint8)

        # 按 fill_fraction 决定多少页含活数据，其余已映射页是空洞页。
        # 活页聚簇 + 随机散布混合，模拟分配器 arena 的区段式布局。
        live_target = int(round(n * cfg.fill_fraction))
        live_target = max(1, min(n, live_target))
        # 70% 的活页聚成若干簇（顺序分配的区段），30% 随机散布
        clustered = int(live_target * 0.7)
        n_clusters = max(1, clustered // max(1, n // 40))
        idx = np.empty(0, dtype=np.int64)
        # 先放聚簇
        centers = self.rng.integers(0, max(1, n), size=n_clusters)
        widths = np.clip(clustered // n_clusters, 1, n)
        lo = np.clip(centers - widths // 2, 0, n - 1)
        hi = np.minimum(lo + widths, n)
        idx = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)])
        idx = np.unique(idx)[:clustered]
        # 再随机补足
        if len(idx) < live_target:
            rest = self.rng.choice(
                np.setdiff1d(np.arange(n), idx, assume_unique=False),
                size=live_target - len(idx),
                replace=False,
            )
            idx = np.concatenate([idx, rest])
        self.state[idx] = PAGE_LIVE
        # 其余位置：60% 为已映射空洞页，40% 保持未映射
        free_mask = self.state == PAGE_FREE
        free_idx = np.flatnonzero(free_mask)
        holes = free_idx[self.rng.random(len(free_idx)) < 0.6]
        self.state[holes] = PAGE_HOLE
        # 活页优先级最高，确保不被上面的步骤覆盖
        self.state[idx] = PAGE_LIVE

        self.n_live = int(np.count_nonzero(self.state == PAGE_LIVE))
        self.n_hole = int(np.count_nonzero(self.state == PAGE_HOLE))
        self.n_mapped = self.n_live + self.n_hole

    # ------------------------------------------------------------------ #
    # 写操作：每个 op 返回它弄脏的基础页号数组（只含已映射页）
    # ------------------------------------------------------------------ #
    def dirty_pages_uniform(self, count: int) -> np.ndarray:
        """均匀随机写：在所有已映射页里均匀挑 count 个页（小键=1页的写法）。"""
        mapped = np.flatnonzero(self.state != PAGE_FREE)
        pick = self.rng.choice(mapped, size=min(count, len(mapped)), replace=True)
        return np.unique(pick)

    def dirty_pages_hot(self, total_writes: int, hot_frac: float = 0.05,
                        hot_share: float = 0.8) -> np.ndarray:
        """热键写：hot_frac 的页吃掉 hot_share 的写入，其余页分剩下的。"""
        mapped = np.flatnonzero(self.state != PAGE_FREE)
        n_hot = max(1, int(len(mapped) * hot_frac))
        hot = mapped[:n_hot]
        cold = mapped[n_hot:]
        n_hot_w = int(total_writes * hot_share)
        n_cold_w = total_writes - n_hot_w
        parts = [self.rng.choice(hot, n_hot_w, replace=True)]
        if len(cold) and n_cold_w > 0:
            parts.append(self.rng.choice(cold, n_cold_w, replace=True))
        return np.unique(np.concatenate(parts))

    def dirty_pages_sequential(self, n_pages_touch: int, offset: int = 0) -> np.ndarray:
        """顺序写（批量导入）：从某偏移开始弄脏一段连续页。"""
        n = len(self.state)
        start = offset % max(1, n)
        end = min(n, start + n_pages_touch)
        touched = np.arange(start, end)
        return touched[self.state[touched] != PAGE_FREE]

    def dirty_pages_large_keys(self, n_keys: int, key_pages_mean: float) -> np.ndarray:
        """大键值写：每个键跨几何分布的连续多页（均值 key_pages_mean 页）。"""
        mapped = np.flatnonzero(self.state != PAGE_FREE)
        out = []
        for _ in range(n_keys):
            span = max(1, int(self.rng.geometric(1.0 / max(1.0, key_pages_mean))))
            base = int(self.rng.integers(0, max(1, len(mapped))))
            take = mapped[base:base + span]
            out.append(take)
        if not out:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(out))

    # ------------------------------------------------------------------ #
    # 删键 / 过期：把活页降级成空洞页（页不归还，仍可被写脏）
    # ------------------------------------------------------------------ #
    def delete_random(self, n_pages: int) -> None:
        live = np.flatnonzero(self.state == PAGE_LIVE)
        if len(live) == 0:
            return
        victims = self.rng.choice(live, min(n_pages, len(live)), replace=False)
        self.state[victims] = PAGE_HOLE
        self.n_live -= len(victims)
        self.n_hole += len(victims)

    def live_page_indices(self) -> np.ndarray:
        return np.flatnonzero(self.state == PAGE_LIVE)

    def expand_to_huge(self, base_pages: np.ndarray) -> np.ndarray:
        """基础页脏集合 -> 开大页时实际被复制的全部基础页（整个大页，向量化）。"""
        if self.cfg.huge_size <= 0 or len(base_pages) == 0:
            return base_pages
        hp = self.cfg.huge_size // self.cfg.page_size
        n = len(self.state)
        touched_huge = np.unique(base_pages // hp)
        offsets = np.arange(hp, dtype=np.int64)
        expanded = (touched_huge[:, None] * hp + offsets[None, :]).ravel()
        expanded = expanded[expanded < n]
        # 大页内未映射页不占物理帧，复制只针对已映射页
        return expanded[self.state[expanded] != PAGE_FREE]


class KeySpace:
    """逐键模型：真实跟踪每个键的字节区间，验证“小键挤页、大键跨页、空洞”。

    只适合小竞技场（默认演示 256 MiB），大场景请用 PageArena。
    """

    def __init__(self, cfg: ArenaConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        ps = cfg.page_size
        n = cfg.n_pages
        self.state = np.zeros(n, dtype=np.uint8)
        # 每个键: (起始字节, 长度, 所在主页号)
        self.keys: list[tuple[int, int, int]] = []
        self._bump = 0  # 模拟 bump/区段分配器的游标
        self._free: list[tuple[int, int]] = []  # 删除后归还的字节区间

    def _alloc(self, size: int) -> int:
        # 优先复用空洞区间（best-fit 简化版），否则从游标往后推
        if self._free:
            best = min(
                ((i, f) for i, f in enumerate(self._free) if f[1] >= size),
                key=lambda t: t[1][1],
                default=None,
            )
            if best is not None:
                i, (start, cap) = best
                del self._free[i]
                if cap - size >= 64:
                    self._free.append((start + size, cap - size))
                return start
        start = self._bump
        self._bump += size
        return start

    def populate(self, small_bytes: tuple = (32, 256), large_bytes: tuple = (8 * 1024, 64 * 1024),
                 large_ratio: float = 0.05) -> None:
        """按目标数据量造初始键集合：95% 小键挤页，5% 大键跨页。"""
        target = self.cfg.data_bytes
        total = 0
        ps = self.cfg.page_size
        while total < target:
            if self.rng.random() < large_ratio:
                size = int(self.rng.integers(*large_bytes))
            else:
                size = int(self.rng.integers(*small_bytes))
            start = self._alloc(size)
            if start + size > self.cfg.mapped_bytes:
                break
            self.keys.append((start, size, start // ps))
            self.state[pages_for_span(start, size, ps)] = PAGE_LIVE
            total += size

    def write_key(self, key_index: int, new_size: int | None = None) -> np.ndarray:
        """改写一个键，返回弄脏的页。

        - 原地改且仍装得下：只弄脏覆盖页；
        - 变大装不下：释放旧区间（留空洞），分配新区间，新旧覆盖页都弄脏
          （真实 Redis 改大对象也是重新分配，旧页变空洞）。
        """
        ps = self.cfg.page_size
        start, size, _ = self.keys[key_index]
        new_size = size if new_size is None else new_size
        dirty = pages_for_span(start, size, ps)
        if new_size <= size:
            return np.unique(dirty)
        # 变大：旧区间释放为空洞，重新分配
        self._free.append((start, size))
        ns = self._alloc(new_size)
        self.keys[key_index] = (ns, new_size, ns // ps)
        dirty = np.concatenate([dirty, pages_for_span(ns, new_size, ps)])
        touched = np.unique(dirty)
        touched = touched[touched < len(self.state)]
        self.state[touched] = np.maximum(self.state[touched], PAGE_HOLE)
        self.state[pages_for_span(ns, new_size, ps)] = PAGE_LIVE
        return touched

    def delete_key(self, key_index: int) -> None:
        start, size, _ = self.keys.pop(key_index)
        self._free.append((start, size))
        touched = pages_for_span(start, size, self.cfg.page_size)
        touched = touched[touched < len(self.state)]
        # 页内还有别的键时仍是活页，这里简化：整页降为空洞（统计模型按页粒度本就如此）
        self.state[touched] = PAGE_HOLE

    def random_key_indices(self, count: int) -> np.ndarray:
        return self.rng.integers(0, len(self.keys), size=count)

    def expand_to_huge(self, base_pages: np.ndarray) -> np.ndarray:
        if self.cfg.huge_size <= 0 or len(base_pages) == 0:
            return base_pages
        hp = self.cfg.huge_size // self.cfg.page_size
        starts = np.unique(base_pages // hp) * hp
        n = len(self.state)
        return np.concatenate([np.arange(s, min(n, s + hp)) for s in starts])

    @property
    def n_live(self) -> int:
        return int(np.count_nonzero(self.state == PAGE_LIVE))

    @property
    def n_hole(self) -> int:
        return int(np.count_nonzero(self.state == PAGE_HOLE))

    @property
    def n_mapped(self) -> int:
        return int(np.count_nonzero(self.state != PAGE_FREE))
