"""
键值数据在内存页上的落位。

真实键值库（Redis / memcached / RocksDB memtable）的分配规律：

  小键值  按 size class 分槽（jemalloc 的 8/16/32/.../256 B 档），
          一个页里塞很多个对象，对象不跨页。写一个键 = 脏 1 页。
  大键值  超过 size class 上限后走大块分配，按页对齐、连续跨页。
          写一个键 = 脏 ceil(size/page) 页。
  删除    小对象把槽位还回 size class，槽位可被后来的键复用；
          大对象把整段页还回 free list，形成空洞。
  过期    与删除相同，只是由后台惰性/定期淘汰触发。

本模块给出两种视图：

  1. `KVStore`   真正逐个键分配，用于可视化与页级 Monte Carlo；
  2. `LayoutStats` 提炼出的统计量，供 30 GiB 级别的解析模型直接使用。

贯穿全文的三个结论：
  * 有效页利用率 u = 已用字节 / 页容量，通常 0.75~0.9，空洞越多 u 越低；
  * 一次写弄脏的页数 d 的分布：绝大多数是 1，大键值是 2~N；
  * 空洞占着的页不会归还操作系统，所以"页数"只增不减 —— 它决定了 COW 的上界。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SMALL_CLASSES = np.array(
    [8, 16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256], dtype=np.int64
)
SMALL_MAX = int(SMALL_CLASSES[-1])


@dataclass
class SizeDistribution:
    """默认贴合"大量小键值 + 少量大键值"的缓存业务。"""
    median_small: float = 96.0
    sigma_small: float = 0.85
    p_large: float = 0.01
    large_min: int = 4096
    large_alpha: float = 1.5       # Pareto 形状参数，越小尾越重
    large_cap: int = 8 << 20

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        is_large = rng.random(n) < self.p_large
        sizes = np.empty(n, dtype=np.int64)
        ns = int((~is_large).sum())
        if ns:
            vals = rng.lognormal(np.log(self.median_small), self.sigma_small, ns)
            sizes[~is_large] = np.clip(np.round(vals), 8, SMALL_MAX).astype(np.int64)
        nl = n - ns
        if nl:
            u = rng.random(nl)
            vals = self.large_min / np.power(u, 1.0 / self.large_alpha)
            sizes[is_large] = np.clip(np.round(vals), self.large_min, self.large_cap).astype(np.int64)
        return sizes


def class_of(sizes: np.ndarray) -> np.ndarray:
    """小对象返回 size-class 下标，大对象返回 -1。"""
    idx = np.searchsorted(SMALL_CLASSES, sizes, side="left")
    idx = np.clip(idx, 0, len(SMALL_CLASSES) - 1)
    idx[sizes > SMALL_MAX] = -1
    return idx


# ---------------------------------------------------------------------------
# 逐个键的存储模型
# ---------------------------------------------------------------------------
class KVStore:
    """
    逐个键维护落位。所有数组按"键 id"索引，删除只把 start 置 -1，
    空间留给后续插入复用（与真实分配器一致）。

    典型用法::

        st = KVStore(page_size=4096, rng=rng)
        st.bulk_insert(sizes)          # 初次装载
        st.delete(ids); st.insert(sizes2)
        st.dirty_pages(key_id)         # [start, start+npages)
    """

    def __init__(self, page_size: int, rng: Optional[np.random.Generator] = None):
        self.page_size = int(page_size)
        self.rng = rng or np.random.default_rng(0)
        self.key_size = np.empty(0, dtype=np.int64)
        self.key_start = np.empty(0, dtype=np.int64)
        self.key_npages = np.empty(0, dtype=np.int64)
        self.key_class = np.empty(0, dtype=np.int64)
        self.page_used = np.empty(0, dtype=np.int64)
        self.page_live = np.empty(0, dtype=np.int64)
        self.free_slot_pages: List[int] = []
        self.free_large: List[Tuple[int, int]] = []
        self.n_deleted = 0

    # ---------- 只读属性 ----------
    @property
    def n_keys(self) -> int:
        return int(self.key_start.size)

    @property
    def n_pages(self) -> int:
        return int(self.page_used.size)

    @property
    def n_live(self) -> int:
        return int((self.key_start >= 0).sum())

    @property
    def capacity_bytes(self) -> int:
        return self.n_pages * self.page_size

    @property
    def used_bytes(self) -> int:
        return int(self.page_used.sum())

    @property
    def util(self) -> float:
        cap = self.capacity_bytes
        return self.used_bytes / cap if cap else 0.0

    def data_bytes(self) -> int:
        live = self.key_start >= 0
        return int(self.key_size[live].sum())

    # ---------- 写入 ----------
    def bulk_insert(self, sizes: Sequence[int]) -> np.ndarray:
        """批量装入，返回新键 id。小对象按 class 紧密装页，大对象连续跨页。"""
        sizes = np.asarray(sizes, dtype=np.int64)
        if sizes.size == 0:
            return np.empty(0, dtype=np.int64)
        ids = np.arange(self.n_keys, self.n_keys + sizes.size, dtype=np.int64)
        cls = class_of(sizes)
        ps = self.page_size

        # 先按 class 计算每类需要的连续页段
        starts = np.full(sizes.size, -1, dtype=np.int64)
        npages = np.zeros(sizes.size, dtype=np.int64)

        for ci, csize in enumerate(SMALL_CLASSES):
            sel = np.flatnonzero(cls == ci)
            if sel.size == 0:
                continue
            csize = int(csize)
            spp = max(1, ps // csize)
            base = self.n_pages
            need = (sel.size + spp - 1) // spp
            local = np.arange(sel.size, dtype=np.int64)
            starts[sel] = base + local // spp
            npages[sel] = 1
            self.page_used = np.concatenate([self.page_used, np.zeros(need, dtype=np.int64)])
            self.page_live = np.concatenate([self.page_live, np.zeros(need, dtype=np.int64)])
            np.add.at(self.page_used, starts[sel], csize)
            np.add.at(self.page_live, starts[sel], 1)
            used_last = sel.size % spp
            if used_last:
                last = base + need - 1
                self.free_slot_pages.extend([last] * (spp - used_last))

        large = np.flatnonzero(cls < 0)
        if large.size:
            npg = np.maximum(1, (sizes[large] + ps - 1) // ps)
            base = self.n_pages
            starts[large] = base + np.concatenate(([0], np.cumsum(npg)[:-1]))
            npages[large] = npg
            total = int(npg.sum())
            self.page_used = np.concatenate([self.page_used, np.zeros(total, dtype=np.int64)])
            self.page_live = np.concatenate([self.page_live, np.zeros(total, dtype=np.int64)])
            for i, kid in enumerate(large):
                s = int(starts[kid]); k = int(npg[i])
                self.page_used[s:s + k] = ps
                self.page_live[s:s + k] += 1
                rem = int(sizes[kid]) - (k - 1) * ps
                self.page_used[s + k - 1] = rem

        self.key_size = np.concatenate([self.key_size, sizes])
        self.key_start = np.concatenate([self.key_start, starts])
        self.key_npages = np.concatenate([self.key_npages, npages])
        self.key_class = np.concatenate([self.key_class, cls])
        return ids

    def delete(self, ids: Sequence[int]) -> None:
        ids = np.asarray(list(ids), dtype=np.int64)
        if ids.size == 0:
            return
        ids = ids[(ids >= 0) & (ids < self.n_keys)]
        ids = ids[self.key_start[ids] >= 0]
        for kid in ids:
            s = int(self.key_start[kid]); k = int(self.key_npages[kid])
            cls = int(self.key_class[kid])
            if k == 1 and cls >= 0:
                self.page_used[s] = max(0, int(self.page_used[s]) - int(SMALL_CLASSES[cls]))
                self.page_live[s] -= 1
                self.free_slot_pages.append(s)
            else:
                self.page_used[s:s + k] = 0
                self.page_live[s:s + k] = 0
                self.free_large.append((s, k))
            self.key_start[kid] = -1
            self.n_deleted += 1

    def insert(self, sizes: Sequence[int]) -> np.ndarray:
        """插入新值，优先复用删除留下的槽位/空洞。返回落在旧页上的键 id。"""
        sizes = np.asarray(list(sizes), dtype=np.int64)
        ids = np.arange(self.n_keys, self.n_keys + sizes.size, dtype=np.int64)
        cls = class_of(sizes)
        ps = self.page_size

        self.key_size = np.concatenate([self.key_size, sizes])
        self.key_class = np.concatenate([self.key_class, cls])
        start = np.full(sizes.size, -1, dtype=np.int64)
        npages = np.zeros(sizes.size, dtype=np.int64)
        reused = np.zeros(sizes.size, dtype=bool)

        slots = self.free_slot_pages
        ranges = self.free_large
        for i, sz in enumerate(sizes):
            ci = int(cls[i])
            if sz <= SMALL_MAX and slots:
                p = slots.pop()
                start[i] = p
                npages[i] = 1
                self.page_used[p] += int(SMALL_CLASSES[ci])
                self.page_live[p] += 1
                reused[i] = True
                continue
            need = 1 if sz <= SMALL_MAX else int((sz + ps - 1) // ps)
            placed = False
            for ri in range(len(ranges) - 1, -1, -1):
                s, k = ranges[ri]
                if k >= need:
                    start[i] = s
                    npages[i] = need
                    self.page_used[s:s + need] = ps
                    self.page_live[s:s + need] += 1
                    rem = sz - (need - 1) * ps
                    self.page_used[s + need - 1] = min(ps, rem)
                    if k > need:
                        ranges[ri] = (s + need, k - need)
                    else:
                        ranges.pop(ri)
                    reused[i] = True
                    placed = True
                    break
            if placed:
                continue
            base = self.n_pages
            start[i] = base
            npages[i] = need
            self.page_used = np.concatenate([self.page_used, np.zeros(need, dtype=np.int64)])
            self.page_live = np.concatenate([self.page_live, np.zeros(need, dtype=np.int64)])
            if sz <= SMALL_MAX:
                csize = int(SMALL_CLASSES[ci])
                self.page_used[base] = csize
                self.page_live[base] = 1
                spp = max(1, ps // csize)
                slots.extend([int(base)] * (spp - 1))
            else:
                self.page_used[base:base + need] = ps
                self.page_live[base:base + need] += 1
                self.page_used[base + need - 1] = sz - (need - 1) * ps

        self.key_start = np.concatenate([self.key_start, start])
        self.key_npages = np.concatenate([self.key_npages, npages])
        return ids[reused]

    # ---------- 查询 ----------
    def dirty_pages(self, kid: int) -> Tuple[int, int]:
        """写一个键会弄脏的页区间 [start, end)；已删除返回 (0, 0)。"""
        s = int(self.key_start[kid])
        if s < 0:
            return 0, 0
        return s, s + int(self.key_npages[kid])

    def key_page_ids(self, kid: int) -> np.ndarray:
        s, e = self.dirty_pages(kid)
        return np.arange(s, e, dtype=np.int64)

    def occupancy_hist(self) -> Dict[str, float]:
        frac = self.page_used / self.page_size
        return {
            "full": float((frac >= 0.99).mean()),
            "partial": float(((frac > 0.01) & (frac < 0.99)).mean()),
            "empty": float((frac <= 0.01).mean()),
        }


# ---------------------------------------------------------------------------
# 统计视图（给解析模型用）
# ---------------------------------------------------------------------------
@dataclass
class LayoutStats:
    page_size: int
    n_pages: int
    data_bytes: int
    used_bytes: int
    util: float
    small_share: float
    large_share: float
    avg_dirty_pages: float
    dirty_hist: Dict[int, float]
    large_pages_share: float
    hole_page_frac: float
    full_page_frac: float
    pages_per_mib: float

    def as_dict(self) -> Dict:
        return {
            "page_size": self.page_size,
            "n_pages": self.n_pages,
            "data_bytes": self.data_bytes,
            "used_bytes": self.used_bytes,
            "util": round(self.util, 4),
            "small_share": round(self.small_share, 4),
            "large_share": round(self.large_share, 4),
            "avg_dirty_pages": round(self.avg_dirty_pages, 3),
            "dirty_hist": {str(k): round(v, 5) for k, v in sorted(self.dirty_hist.items())},
            "large_pages_share": round(self.large_pages_share, 4),
            "hole_page_frac": round(self.hole_page_frac, 4),
            "full_page_frac": round(self.full_page_frac, 4),
            "pages_per_mib": round(self.pages_per_mib, 2),
        }


def layout_stats(st: KVStore) -> LayoutStats:
    ps = st.page_size
    live = st.key_start >= 0
    used = int(st.page_used.sum())
    total = int(st.key_size[live].sum()) if live.any() else 0
    small_bytes = int(st.key_size[live & (st.key_size <= SMALL_MAX)].sum())
    d = np.where(st.key_size <= SMALL_MAX, 1, st.key_npages)[live]
    if d.size == 0:
        d = np.array([1])
    uniq, cnt = np.unique(d, return_counts=True)
    hist = {int(u): float(c) / d.size for u, c in zip(uniq, cnt)}
    large_pages = int(st.key_npages[live & (st.key_npages > 1)].sum())
    frac = st.page_used / ps
    return LayoutStats(
        page_size=ps,
        n_pages=st.n_pages,
        data_bytes=total,
        used_bytes=used,
        util=used / st.capacity_bytes if st.n_pages else 0.0,
        small_share=small_bytes / total if total else 0.0,
        large_share=1 - small_bytes / total if total else 0.0,
        avg_dirty_pages=float(d.mean()),
        dirty_hist=hist,
        large_pages_share=large_pages / st.n_pages if st.n_pages else 0.0,
        hole_page_frac=float(((frac > 0.01) & (frac < 0.99)).mean()),
        full_page_frac=float((frac >= 0.99).mean()),
        pages_per_mib=(2 ** 20) / ps,
    )


def pages_for_bytes(data_bytes: float, page_size: int, util: float) -> int:
    """按有效利用率把逻辑数据量换算成页数。"""
    return int(np.ceil(data_bytes / (page_size * util)))
