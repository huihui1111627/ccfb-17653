"""
平台画像：把"页大小 / COW 拷贝代价 / fork 停顿 / 大页代价"这些
与操作系统相关的常数集中在一处。

数据来源分两类，字段里用注释标出：

  [实测]  calib/calibrate.py 在本机（macOS 26 / arm64 / 16 KiB 页）测得
  [文献]  Linux 4 KiB 页 / 透明大页的典型值，按保守区间取中位数

所有时间单位都在字段名里写清楚（_us / _ms），避免单位换算错误。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict


@dataclass(frozen=True)
class PageProfile:
    key: str
    label: str
    base_page: int                 # 基页字节数
    huge_page: int                 # 大页字节数，0 表示该平台没有可用大页
    cow_us_p50: float              # 触发一次基页 COW 的停顿中位数（微秒）
    cow_us_p99: float
    plain_write_us: float          # 不触发 COW 的普通写延迟（微秒）
    fork_ms_base: float            # fork 固定开销（毫秒）
    fork_ms_per_mpage: float       # 每百万页 page table 复制开销（毫秒）
    thp_copy_us_p50: float         # 复制一个 2 MiB 大页的停顿时延
    thp_copy_us_p99: float
    thp_stall_prob: float          # 大页分配触发内存规整/压缩的概率
    thp_stall_ms: float            # 触发时额外停顿
    pte_bytes_per_page: float      # 每个基页对应的页表项字节数（COW 时页表也要复制）
    scan_gib_s: float              # 子进程顺序扫描吞吐（仅用于自检，不决定快照时长）
    note: str = ""

    @property
    def block_pages(self) -> int:
        """一个大页包含多少个基页。"""
        if not self.huge_page:
            return 1
        return max(1, self.huge_page // self.base_page)

    def as_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Linux 4 KiB：生产环境的默认画像
# ---------------------------------------------------------------------------
# cow_us_p50：4 KiB 页复制（~0.15us memcpy）+ 缺页/分配/page-fault 记账 ≈ 1.2~1.5us
# cow_us_p99：分配器竞争、跨 NUMA、mmap_lock 读侧抖动后的尾部
# fork_ms_per_mpage：dup_mmap 要复制 PTE（512 项/页，8B/项），30 GiB ≈ 7.9M 页
#                   ≈ 60 MiB 页表 + VMA 遍历，实测社区报告 20~80 ms
LINUX_4K = PageProfile(
    key="linux-4k",
    label="Linux 4 KiB 页",
    base_page=4096,
    huge_page=2 * 1024 * 1024,
    cow_us_p50=1.3,
    cow_us_p99=3.4,
    plain_write_us=0.12,
    fork_ms_base=0.9,
    fork_ms_per_mpage=4.0,
    thp_copy_us_p50=45.0,
    thp_copy_us_p99=130.0,
    thp_stall_prob=0.02,
    thp_stall_ms=1.8,
    pte_bytes_per_page=8.0,
    scan_gib_s=8.0,
    note="[文献] 4 KiB 基页；THP 2 MiB；fork 复制页表",
)

# ---------------------------------------------------------------------------
# 本机：macOS / arm64 / 16 KiB 页（实测）
# ---------------------------------------------------------------------------
DARWIN_16K = PageProfile(
    key="darwin-16k",
    label="macOS 16 KiB 页（本机）",
    base_page=16384,
    huge_page=0,
    cow_us_p50=2.08,
    cow_us_p99=5.72,
    plain_write_us=0.17,
    fork_ms_base=0.35,
    fork_ms_per_mpage=0.0,
    thp_copy_us_p50=0.0,
    thp_copy_us_p99=0.0,
    thp_stall_prob=0.0,
    thp_stall_ms=0.0,
    pte_bytes_per_page=8.0,
    scan_gib_s=32.0,
    note="[实测] 16 KiB 页；fork 时页表惰性复制，停顿与数据量基本无关",
)

# 本机用于模拟 2 MiB 大页语义的画像：写一个字节 -> 强制触碰 128 个 16 KiB 页
# 让内核做 128 次 COW，等价于复制整个 2 MiB 块。拷贝代价用实测 memcpy(2 MiB)
# 30.3 us + 128 次缺页开销 ≈ 34 us。
DARWIN_16K_HUGE_EMU = PageProfile(
    key="darwin-16k-huge-emu",
    label="macOS 16 KiB 页 + 2 MiB 大页（模拟）",
    base_page=16384,
    huge_page=2 * 1024 * 1024,
    cow_us_p50=2.08,
    cow_us_p99=5.72,
    plain_write_us=0.17,
    fork_ms_base=0.35,
    fork_ms_per_mpage=0.0,
    thp_copy_us_p50=237.0,       # [实测] 512 MiB 数据、256 个 2 MiB 块的 COW 停顿中位数
    thp_copy_us_p99=300.0,       # [实测]
    thp_stall_prob=0.0,          # macOS 不支持大页，不存在内存规整
    thp_stall_ms=0.0,
    pte_bytes_per_page=8.0,
    scan_gib_s=32.0,
    note="[实测] 大页语义由 bench 的 huge 模式强制拷贝整个 2 MiB 块模拟；"
         "注意这是 128 次 16K 缺页的合计，Linux 原生 THP 实测通常 45~150 us",
)

PROFILES: Dict[str, PageProfile] = {
    p.key: p for p in (LINUX_4K, DARWIN_16K, DARWIN_16K_HUGE_EMU)
}


def get(key: str) -> PageProfile:
    if key not in PROFILES:
        raise KeyError(f"未知平台画像 {key!r}，可选：{sorted(PROFILES)}")
    return PROFILES[key]
