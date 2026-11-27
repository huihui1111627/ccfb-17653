"""核心机制单元测试（小竞技场，秒级跑完）：
  python3 -m cowkit.tests
覆盖：落页规则、大页展开、COW 单调性、闭式公式一致性、OOM 判定、缺页概率边界。
"""

from __future__ import annotations

import numpy as np

from .pages import ArenaConfig, PageArena, KeySpace, pages_for_span, THP_2M, PAGE_LIVE
from .workloads import WorkloadSpec, expected_dirty_fraction
from .engine import simulate_snapshot, LatencyModel
from .analytic import AnalyticInput, copied_pages_curve

MIB = 1024 ** 2
GIB = 1024 ** 3
PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name}  {detail}")


def test_pages_for_span():
    print("[pages_for_span]")
    check("单字节落在单页", (pages_for_span(0, 1, 4096) == [0]).all())
    check("跨页边界覆盖两页", (pages_for_span(4090, 12, 4096) == [0, 1]).all())
    check("大对象跨 4 页", len(pages_for_span(10, 3 * 4096, 4096)) == 4)
    check("零长度无页", len(pages_for_span(0, 0, 4096)) == 0)


def test_arena_layout():
    print("[PageArena 布局]")
    cfg = ArenaConfig(data_bytes=64 * MIB, page_size=4096, fill_fraction=0.85, seed=1)
    a = PageArena(cfg)
    check("活页+空洞页=已映射", a.n_live + a.n_hole == a.n_mapped)
    check("已映射不超过总页", a.n_mapped <= len(a.state))
    check("活页占比接近 fill_fraction",
          abs(a.n_live / len(a.state) - 0.85) < 0.05,
          f"{a.n_live/len(a.state):.3f}")
    check("均匀写脏页数 ≤ 写入数", len(a.dirty_pages_uniform(1000)) <= 1000)
    seq = a.dirty_pages_sequential(500)
    # 顺序写返回区间内的已映射页（未映射页被跳过），应严格递增且跨度覆盖该区间
    check("顺序写页号严格递增", bool(np.all(np.diff(seq) > 0)))
    check("顺序写覆盖连续区间", (seq.max() - seq.min() + 1) >= len(seq))


def test_huge_expand():
    print("[THP 大页展开]")
    check("THP_2M 是 2MiB 而非 2GiB", THP_2M == 2 * MIB)
    cfg = ArenaConfig(data_bytes=256 * MIB, page_size=4096,
                      huge_size=THP_2M, fill_fraction=0.9, seed=3)
    a = PageArena(cfg)
    hp = THP_2M // 4096
    check("每大页含 512 基础页", hp == 512)
    base = a.dirty_pages_uniform(200)
    exp = a.expand_to_huge(base)
    check("大页展开后页数 ≥ 基础脏页", len(exp) >= len(base))
    # 展开后的页必须恰好由完整大页组成（每个大页最多 hp 个基础页）
    big = np.unique(exp // hp)
    check("展开结果按大页对齐",
          all((exp // hp == h).sum() <= hp for h in big))
    check("单字节写也最多复制一个大页",
          len(a.expand_to_huge(base[:1])) <= hp)


def test_keyspace():
    print("[KeySpace 逐键]")
    cfg = ArenaConfig(data_bytes=8 * MIB, page_size=4096, fill_fraction=0.85, seed=4)
    ks = KeySpace(cfg)
    ks.populate()
    check("确实造了键", len(ks.keys) > 100)
    n_live_before = ks.n_live
    d = ks.write_key(5, new_size=4096 * 10)
    check("改大键弄脏多个页", len(d) >= 10)
    n_keys = len(ks.keys)
    ks.delete_key(0)
    check("删键后键数减一", len(ks.keys) == n_keys - 1)


def test_cow_monotonic_and_oom():
    print("[COW 单调增长与 OOM]")
    cfg = ArenaConfig(data_bytes=32 * MIB, page_size=4096, fill_fraction=0.9, seed=5)
    a = PageArena(cfg)
    spec = WorkloadSpec("uniform", write_rate=2000, duration=20, seed=5)
    # 给一个极小的 RAM，必然 OOM
    r = simulate_snapshot(a, spec, ram_bytes=20 * MIB, n_buckets=20)
    check("复制页数单调不减", bool(np.all(np.diff(r.copied_pages) >= 0)))
    check("footprint 单调不减", bool(np.all(np.diff(r.footprint_bytes) >= 0)))
    check("小 RAM 触发 OOM", r.oom_at is not None)
    check("故障率在 [0,1]", float(r.fault_req_rate.min()) >= 0 and float(r.fault_req_rate.max()) <= 1.0)
    # 大 RAM 不 OOM
    r2 = simulate_snapshot(a, spec, ram_bytes=4 * GIB, n_buckets=20)
    check("充足 RAM 不 OOM", r2.oom_at is None)


def test_thp_fault_decay():
    print("[THP 缺页风暴衰减]")
    # 128MiB ≈ 64 个 2MiB 大页容器；60 写/秒下风暴跨越多桶、可观测衰减
    cfg = ArenaConfig(data_bytes=128 * MIB, page_size=4096,
                      huge_size=THP_2M, fill_fraction=0.9, seed=6)
    a = PageArena(cfg)
    spec = WorkloadSpec("uniform", write_rate=60, duration=30, seed=6)
    r = simulate_snapshot(a, spec, ram_bytes=4 * GIB,
                          model=LatencyModel(base_median_us=100, base_p99_us=260),
                          n_buckets=30)
    first = float(r.fault_req_rate[0])
    late = float(r.fault_req_rate[-5:].mean())
    check("大页故障率首桶高", first > 0.5, f"first={first:.2f}")
    check("大页故障率后期衰减", late < first, f"late={late:.2f}")
    check("故障率在 [0,1]", 0.0 <= float(r.fault_req_rate.min())
          and float(r.fault_req_rate.max()) <= 1.0)


def test_analytic_vs_sim():
    print("[闭式公式 vs 仿真]")
    cfg = ArenaConfig(data_bytes=64 * MIB, page_size=4096, fill_fraction=0.9, seed=7)
    a = PageArena(cfg)
    writes = 20000
    spec = WorkloadSpec("uniform", write_rate=writes / 20, duration=20, seed=7)
    r = simulate_snapshot(a, spec, ram_bytes=4 * GIB, n_buckets=40)
    ai = AnalyticInput(n_mapped_pages=a.n_mapped, page_size=4096,
                       writes_total=writes, workload="uniform")
    pred = copied_pages_curve(ai, [writes])[0]
    err = abs(pred - r.copied_pages[-1]) / max(1, r.copied_pages[-1])
    check("均匀随机写公式误差 < 10%", err < 0.10, f"err={err:.3f}")
    # 期望脏页比例的 sanity
    f = expected_dirty_fraction("uniform", writes, a.n_mapped)
    check("脏页比例在 (0,1)", 0 < f < 1)


def main():
    for fn in (test_pages_for_span, test_arena_layout, test_huge_expand,
               test_keyspace, test_cow_monotonic_and_oom,
               test_thp_fault_decay, test_analytic_vs_sim):
        fn()
    print(f"\n{PASS} 通过, {FAIL} 失败")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
