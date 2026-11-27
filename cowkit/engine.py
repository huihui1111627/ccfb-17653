"""快照持久化期间的 COW 逐拍仿真引擎（需求 2/4 的核心）。

仿真内容
========
* fork 瞬间：父进程停顿 = fork 成本（复制页表，约 8~15 us/ GiB，随已映射内存线性）；
* 快照窗口内每个写请求：
    - 计算它弄脏的基础页；若开 THP，展开成整张大页；
    - 尚未复制过的页 -> 一次缺页异常 -> 内核复制整页 -> 父进程停顿一次；
    - 已复制过的页 -> 不再停顿（这正是热键写后期 COW 曲线变平的原因）；
* 每条请求延迟 = 基础服务延迟（对数正态）+ 缺页复制停顿 + 内存压力停顿；
* 父进程 footprint = 初始 RSS + 已复制字节 + 快照期新分配字节；
  超过物理内存阈值 -> 模拟 OOM 杀进程（对应用户说的“被系统杀掉”）。

输出
====
SnapshotResult：时间曲线、延迟分位数、OOM 事件、以及供热力图用的页状态采样。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .pages import PageArena, KeySpace
from . import workloads as wl


@dataclass
class LatencyModel:
    """延迟参数（单位：秒）。默认值按 Linux 4K 页 / 2M THP 的典型量级标定，
    real_fork 实测会在本机重新标定 base_fault_us 并覆盖这里。"""

    base_median_us: float = 300.0          # 无缺页时请求延迟中位数
    base_p99_us: float = 950.0             # 无缺页时 p99（决定对数正态 sigma）
    base_fault_us: float = 8.0             # 4K 基础页缺页复制停顿
    huge_fault_us: float = 150.0           # 2M 大页缺页复制停顿
    fork_us_per_gib: float = 10_000.0      # fork 页表复制：10ms/GiB（Redis 实测量级）
    pressure_full_at: float = 0.98         # footprint 占物理内存比例 -> 压力停满
    pressure_extra_us: float = 2000.0      # 压力最大时额外停顿
    rng_seed: int = 42

    @property
    def sigma(self) -> float:
        # 对数正态: p99/median = e^(2.326 sigma)
        return math.log(self.base_p99_us / self.base_median_us) / 2.326347874

    @property
    def mu(self) -> float:
        return math.log(self.base_median_us) - self.sigma ** 2 / 2


@dataclass
class SnapshotResult:
    config: dict
    t: np.ndarray = field(default_factory=lambda: np.empty(0))
    writes_cum: np.ndarray = field(default_factory=lambda: np.empty(0))
    copied_pages: np.ndarray = field(default_factory=lambda: np.empty(0))
    copied_huge: np.ndarray = field(default_factory=lambda: np.empty(0))
    footprint_bytes: np.ndarray = field(default_factory=lambda: np.empty(0))
    faults_cum: np.ndarray = field(default_factory=lambda: np.empty(0))
    fault_req_rate: np.ndarray = field(default_factory=lambda: np.empty(0))
    lat_all_us: np.ndarray = field(default_factory=lambda: np.empty(0))
    lat_fault_us: np.ndarray = field(default_factory=lambda: np.empty(0))
    fork_stall_us: float = 0.0
    oom_at: float | None = None
    heatmap: dict = field(default_factory=dict)
    workload: str = ""
    huge_on: bool = False

    def percentile_us(self, q: float) -> float:
        if len(self.lat_all_us) == 0:
            return float("nan")
        return float(np.percentile(self.lat_all_us, q))

    def summary(self) -> dict:
        cp = self.copied_pages[-1] if len(self.copied_pages) else 0
        fp = self.footprint_bytes[-1] if len(self.footprint_bytes) else 0
        return {
            "workload": self.workload,
            "huge_on": self.huge_on,
            "n_writes": int(self.lat_all_us.size),
            "copied_pages_peak": int(cp),
            "copied_bytes_peak": int(cp * self.config["page_size"]),
            "footprint_peak_bytes": int(fp),
            "fork_stall_ms": self.fork_stall_us / 1000.0,
            "p50_us": self.percentile_us(50),
            "p99_us": self.percentile_us(99),
            "p999_us": self.percentile_us(99.9),
            "p9999_us": self.percentile_us(99.99),
            "max_us": float(self.lat_all_us.max()) if self.lat_all_us.size else 0.0,
            "fault_request_pct": 100.0 * len(self.lat_fault_us) / max(1, self.lat_all_us.size),
            "oom": self.oom_at is not None,
            "oom_at_s": self.oom_at,
        }


def _pressure_factor(footprint: float, ram_bytes: int, model: LatencyModel) -> float:
    """footprint 从 80% 到 pressure_full_at 之间，压力停顿线性升到满。"""
    ratio = footprint / max(1, ram_bytes)
    lo, hi = 0.80, model.pressure_full_at
    return float(np.clip((ratio - lo) / (hi - lo), 0.0, 1.0))


def simulate_snapshot(arena, spec: wl.WorkloadSpec, ram_bytes: int,
                      model: LatencyModel | None = None,
                      n_buckets: int = 120,
                      heatmap_cells: int = 2400,
                      new_alloc_rate_bps: float = 0.0) -> SnapshotResult:
    """跑一次完整快照窗口仿真。

    参数
    ----
    arena                 : PageArena 或 KeySpace
    spec                  : 写入模式规格
    ram_bytes             : 单机物理内存（判定压力/OOM）
    n_buckets             : 时间曲线采样桶数
    heatmap_cells         : 热力图降采样后的格子数
    new_alloc_rate_bps    : 快照期新分配字节/秒（新键、键变大）
    """
    model = model or LatencyModel()
    rng = np.random.default_rng(model.rng_seed + spec.seed)
    cfg = arena.cfg
    huge_on = cfg.huge_size > 0
    ps = cfg.page_size
    n_pages = len(arena.state)
    rss0 = arena.n_mapped * ps

    gen = wl.WORKLOADS[spec.name](spec)
    events = list(gen)
    n_writes = len(events)

    copied = np.zeros(n_pages, dtype=bool)
    # 按时间分桶，桶内批量处理脏页（向量化），延迟按请求逐条抽样
    bucket_edges = np.linspace(0, spec.duration, n_buckets + 1)
    t_series, wr_series, cp_series, ch_series, fp_series, flt_series = [], [], [], [], [], []
    fault_rate_series = []
    lat_samples, fault_lat_samples = [], []
    faults_total = 0
    oom_at = None

    # fork 瞬间停顿（复制页表，与已映射内存线性；THP 大幅减少页表项 => 打 4 折）
    fork_factor = 0.4 if huge_on else 1.0
    fork_stall_us = model.fork_us_per_gib * (rss0 / (1024 ** 3)) * fork_factor

    seq_cursor = 0
    bucket_idx = 0
    writes_done = 0

    def bucket_dirty(lo: int, hi: int) -> np.ndarray:
        """把 [lo,hi) 号事件折算成脏基础页集合。"""
        nonlocal seq_cursor
        kinds = [events[i][1]["kind"] for i in range(lo, hi)]
        pages_per = np.array([events[i][1]["pages"] for i in range(lo, hi)], dtype=np.int64)
        if spec.name == "sequential":
            total_pages = int(pages_per.sum())
            out = arena.dirty_pages_sequential(total_pages, seq_cursor)
            seq_cursor += total_pages
            return out
        if spec.name == "hotkey":
            hot_frac = events[lo][1].get("hot_frac", 0.05)
            return arena.dirty_pages_hot(hi - lo, hot_frac=hot_frac)
        if spec.name == "mixed":
            # 小键按均匀单页，大键按跨页几何分布拆开
            kinds_arr = np.array(kinds)
            n_large = int((kinds_arr == "large").sum())
            n_small = (hi - lo) - n_large
            parts = []
            if n_small:
                parts.append(arena.dirty_pages_uniform(n_small))
            if n_large:
                large_mean = float(pages_per[kinds_arr == "large"].mean())
                parts.append(arena.dirty_pages_large_keys(
                    n_large, key_pages_mean=max(1.0, large_mean)))
            return np.unique(np.concatenate(parts)) if parts else np.empty(0, np.int64)
        # uniform
        return arena.dirty_pages_uniform(hi - lo)

    edge_it = 0
    for b in range(n_buckets):
        t_hi = bucket_edges[b + 1]
        lo = edge_it
        while edge_it < n_writes and events[edge_it][0] < t_hi:
            edge_it += 1
        hi = edge_it
        bucket_n = hi - lo

        if bucket_n:
            dirty = bucket_dirty(lo, hi)
            if huge_on:
                # 容器视角：本桶新触碰的大页才是“缺页事件”，一个大页只故障一次
                hp = cfg.huge_size // ps
                touched_huge = np.unique(dirty // hp)
                # 该大页是否已复制：任取其中一个基础页看 copied 标志即可
                sample_page = np.minimum(touched_huge * hp, n_pages - 1)
                already = copied[sample_page]
                new_huge = touched_huge[~already]
                n_new_containers = len(new_huge)
                touched = arena.expand_to_huge(dirty)
            else:
                touched = dirty
                n_new_containers = None
            touched = touched[touched < n_pages]
            new_mask = ~copied[touched]
            new_pages = touched[new_mask]
            n_new = len(new_pages)
            copied[new_pages] = True
            faults_total += n_new

            # ---- 逐条请求延迟抽样 ----
            base = np.exp(rng.normal(model.mu, model.sigma, size=bucket_n))  # us
            # 缺页概率必须在“复制容器”层面算：
            #   基础页模式 -> 新基础页数/请求数；大页模式 -> 新大页数/请求数。
            # （大页一次故障复制 512 个基础页，用基础页数会恒等于 1，是错的。）
            n_fault_events = n_new if n_new_containers is None else n_new_containers
            p_fault = min(1.0, n_fault_events / max(1, bucket_n))
            is_fault = rng.random(bucket_n) < p_fault
            fault_cost = model.huge_fault_us if huge_on else model.base_fault_us
            # 大键一次跨多页：停顿按实际新页数折算（几何分布近似）
            if spec.name == "mixed":
                pp = np.array([events[i][1]["pages"] for i in range(lo, hi)])
                cost_per = np.where(pp > 1, pp * fault_cost, fault_cost)
                base[is_fault] += cost_per[is_fault]
            else:
                base[is_fault] += fault_cost
            lat_samples.append(base)
            fault_lat_samples.append(base[is_fault])
            # 撞缺页的请求占比（0~1），供热力图/延迟对齐视图使用
            fault_rate_series.append(float(is_fault.mean()))
        else:
            fault_rate_series.append(0.0)

        writes_done += bucket_n
        footprint = rss0 + int(copied.sum()) * ps + int(new_alloc_rate_bps * t_hi)
        p = _pressure_factor(footprint, ram_bytes, model)
        if bucket_n and p > 0:
            extra = rng.exponential(model.pressure_extra_us * p, size=bucket_n)
            lat_samples[-1] = lat_samples[-1] + extra
        if oom_at is None and footprint >= ram_bytes * model.pressure_full_at:
            oom_at = float(t_hi)

        t_series.append(t_hi)
        wr_series.append(writes_done)
        cp_series.append(int(copied.sum()))
        if huge_on:
            # 已复制的大页数：已复制基础页所在大页号去重计数
            ch_series.append(int(np.unique(
                copied.nonzero()[0] // (cfg.huge_size // ps)).size))
        else:
            ch_series.append(0)
        fp_series.append(footprint)
        flt_series.append(faults_total)

    lat_all = np.concatenate(lat_samples) if lat_samples else np.empty(0)
    lat_flt = np.concatenate(fault_lat_samples) if fault_lat_samples else np.empty(0)

    # ---- 热力图：把页状态降采样成固定格数，记录 0/25/50/75/100% 五个时刻 ----
    heat = _build_heatmap(arena, copied, n_pages, heatmap_cells,
                          bucket_edges, t_series, cp_series)

    return SnapshotResult(
        config={"page_size": ps, "huge_size": cfg.huge_size,
                "n_mapped_pages": arena.n_mapped, "ram_bytes": ram_bytes,
                "rss0_bytes": rss0, "snapshot_s": spec.duration,
                "write_rate": spec.write_rate},
        t=np.array(t_series), writes_cum=np.array(wr_series),
        copied_pages=np.array(cp_series), copied_huge=np.array(ch_series),
        footprint_bytes=np.array(fp_series), faults_cum=np.array(flt_series),
        fault_req_rate=np.array(fault_rate_series),
        lat_all_us=lat_all, lat_fault_us=lat_flt,
        fork_stall_us=fork_stall_us, oom_at=oom_at,
        heatmap=heat, workload=spec.name, huge_on=huge_on)


def _build_heatmap(arena, copied, n_pages, cells, edges, t_series, cp_series) -> dict:
    """页状态热力图：活页=蓝、空洞=灰、已复制=红（颜色在 dashboard 里映射）。"""
    width = 60
    height = max(1, cells // width)
    grid_n = width * height
    # 把 n_pages 压到 grid_n 个格子，每格记录活页数/空洞数/已复制数
    bins = np.linspace(0, n_pages, grid_n + 1).astype(np.int64)
    state = arena.state
    live = (state == 2)
    hole = (state == 1)

    def aggregate(mask_copied):
        g_live = np.zeros(grid_n)
        g_hole = np.zeros(grid_n)
        g_copy = np.zeros(grid_n)
        for i in range(grid_n):
            a, b = bins[i], bins[i + 1]
            if b <= a:
                continue
            cap = b - a
            g_live[i] = live[a:b].sum() / cap
            g_hole[i] = hole[a:b].sum() / cap
            g_copy[i] = mask_copied[a:b].sum() / cap
        return g_live.reshape(height, width).tolist(), \
               g_hole.reshape(height, width).tolist(), \
               g_copy.reshape(height, width).tolist()

    # 终态 + 四个中间时刻（中间时刻用当前 copied 无法回放，这里给终态与初始态，
    # 中间帧由 dashboard 按复制曲线比例插值生成，避免存五份全量位图）
    live_g, hole_g, copy_g = aggregate(copied)
    _, _, zero_g = aggregate(np.zeros(n_pages, dtype=bool))
    cp_arr = np.array(cp_series, dtype=float)
    copy_frac_curve = (cp_arr / max(1, cp_arr.max())).tolist() if cp_arr.size else []
    return {"width": width, "height": height,
            "live": live_g, "hole": hole_g,
            "copy_final": copy_g, "copy_zero": zero_g,
            "copy_frac_curve": copy_frac_curve}
