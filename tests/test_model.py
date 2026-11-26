"""模型不变量测试。运行：python3 -m pytest tests -q"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sim.heat import groups_for, zipf_groups, _zipf_h
from sim.layout import KVStore, SizeDistribution, SMALL_MAX, class_of, layout_stats
from sim.model import (DIST_HOT, DIST_SEQUENTIAL, DIST_UNIFORM, DIST_ZIPF,
                       Scenario, distinct_touched, evaluate, resolve, simulate)
from sim.pagesim import BitSetSim, BucketSim
from sim.profiles import DARWIN_16K, LINUX_4K, get


# ---------------------------------------------------------------------------
# 热度分组
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1000, 100_000, 900_000, 8_000_000])
@pytest.mark.parametrize("s", [0.9, 1.0, 1.1, 1.3, 1.6])
def test_zipf_groups_normalized(n, s):
    sizes, mass = zipf_groups(n, s)
    assert sizes.sum() == pytest.approx(n, rel=1e-9)
    assert mass.sum() == pytest.approx(1.0, rel=1e-9)
    assert (sizes > 0).all()
    assert (mass >= 0).all()


def test_zipf_groups_match_bruteforce_head():
    n, s = 100_000, 1.2
    sizes, mass = zipf_groups(n, s)
    brute = np.power(np.arange(1, n + 1, dtype=float), -s)
    brute /= brute.sum()
    assert mass[:50].sum() == pytest.approx(brute[:50].sum(), rel=1e-9)


def test_heat_groups_probability_mass_one():
    for dist in (DIST_UNIFORM, DIST_HOT, DIST_ZIPF):
        sc = Scenario(distribution=dist, hot_space=0.002, hot_fraction=0.9)
        sizes, probs = groups_for(sc, 500_000)
        assert (sizes * probs).sum() == pytest.approx(1.0, rel=1e-6)
        assert (probs >= 0).all()
        assert (probs <= 1).all()


# ---------------------------------------------------------------------------
# E(n)
# ---------------------------------------------------------------------------
def test_uniform_matches_closed_form():
    P = 200_000
    sc = Scenario(distribution=DIST_UNIFORM)
    for n in (1_000, 50_000, 400_000, 2_000_000):
        e = distinct_touched(n, P, sc)
        expected = P * (1 - math.exp(-n / P))
        assert e == pytest.approx(expected, rel=1e-3)


def test_sequential_is_min():
    P = 100_000
    sc = Scenario(distribution=DIST_SEQUENTIAL)
    assert distinct_touched(30_000, P, sc) == pytest.approx(30_000)
    assert distinct_touched(300_000, P, sc) == pytest.approx(P)


def test_E_is_monotone_and_bounded():
    sc = Scenario(distribution=DIST_ZIPF)
    P = 300_000
    prev = 0
    for n in (1e3, 1e4, 1e5, 1e6, 1e7):
        e = distinct_touched(n, P, sc)
        assert e >= prev - 1e-6
        assert e <= P * 1.0001
        prev = e


def test_E_never_exceeds_blocks():
    for dist in (DIST_UNIFORM, DIST_HOT, DIST_ZIPF, DIST_SEQUENTIAL):
        sc = Scenario(distribution=dist, locality=0.7)
        P = 123_456
        assert distinct_touched(1e9, P, sc) <= P * 1.0001


# ---------------------------------------------------------------------------
# 峰值上界
# ---------------------------------------------------------------------------
def test_peak_never_exceeds_two_times():
    for huge in (False, True):
        for dist in (DIST_UNIFORM, DIST_HOT, DIST_ZIPF, DIST_SEQUENTIAL):
            for loc in (1.0, 0.3):
                sc = Scenario(distribution=dist, locality=loc, data_gib=10,
                              write_rate=100_000, snapshot_seconds=600,
                              huge_pages=huge)
                r = evaluate(sc, LINUX_4K, pressure=False)
                assert r["peak_ratio"] <= 2.0 + 1e-6


def test_huge_pages_copy_in_quantized_units():
    """大页下复制量按 2 MiB 块计；不足一块的尾部被封顶到数据区大小。"""
    sc = Scenario(data_gib=8, write_rate=20_000, snapshot_seconds=30, huge_pages=True)
    rv, ser, diag = simulate(sc, LINUX_4K, pressure=False)
    assert rv.block_bytes == 2 * 1024 * 1024
    copied = ser.copied_bytes[-1]
    assert copied == pytest.approx(min(ser.copied_blocks[-1] * rv.block_bytes,
                                       rv.resident_bytes))
    assert copied <= rv.resident_bytes + 1e-6    # 永远不超过 2 倍


def test_bigger_data_more_copies_at_same_write_count():
    """同样写入量，数据越大复制比例越低（同样的写分散到更多页上）。"""
    rs = []
    for g in (5, 20, 80):
        sc = Scenario(data_gib=g, write_rate=50_000, snapshot_seconds=60)
        rs.append(evaluate(sc, LINUX_4K, pressure=False)["copied_fraction"])
    assert rs[0] > rs[1] > rs[2]


# ---------------------------------------------------------------------------
# 仿真一致性
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dist", [DIST_UNIFORM, DIST_HOT, DIST_ZIPF, DIST_SEQUENTIAL])
@pytest.mark.parametrize("loc", [1.0, 0.3])
def test_sim_matches_model(dist, loc):
    sc = Scenario(distribution=dist, locality=loc, data_gib=2.0, util=0.85,
                  write_rate=50_000, snapshot_seconds=20)
    an = evaluate(sc, LINUX_4K, pressure=False)
    sim = BitSetSim(sc, seed=3).run().summary()
    assert sim["peak_gib"] == pytest.approx(an["peak_gib"], rel=0.02)


def test_bucket_matches_bitset():
    for dist in (DIST_UNIFORM, DIST_HOT, DIST_ZIPF):
        sc = Scenario(distribution=dist, data_gib=2.0, write_rate=50_000,
                      snapshot_seconds=20)
        a = BitSetSim(sc, seed=1).run().summary()["peak_gib"]
        b = BucketSim(sc, seed=1).run().summary()["peak_gib"]
        assert b == pytest.approx(a, rel=0.03)


# ---------------------------------------------------------------------------
# 布局
# ---------------------------------------------------------------------------
def test_small_keys_dirty_one_page():
    rng = np.random.default_rng(0)
    st = KVStore(4096)
    st.bulk_insert(np.full(10_000, 64))
    for kid in range(100):
        s, e = st.dirty_pages(kid)
        assert e - s == 1


def test_large_key_spans_pages():
    st = KVStore(4096)
    st.bulk_insert([10 * 4096 + 100])
    s, e = st.dirty_pages(0)
    assert e - s == 11


def test_delete_creates_holes_and_reuse():
    rng = np.random.default_rng(2)
    st = KVStore(4096, rng)
    st.bulk_insert(np.full(100_000, 64))
    pages_before = st.n_pages
    st.delete(np.arange(0, 50_000))
    assert st.n_pages == pages_before          # 页不会归还
    reused = st.insert(np.full(20_000, 64))
    assert reused.size > 0                     # 复用了旧页上的槽位


def test_layout_util_exceeds_one_for_slab_internal_waste():
    """按 size class 计入后，占用字节 >= 逻辑字节（slab 碎片）。"""
    rng = np.random.default_rng(0)
    sizes = SizeDistribution(median_small=50, p_large=0.001).sample(200_000, rng)
    st = KVStore(4096)
    st.bulk_insert(sizes)
    assert st.used_bytes >= st.data_bytes()


# ---------------------------------------------------------------------------
# 大页
# ---------------------------------------------------------------------------
def test_huge_pages_worse_or_equal_on_memory():
    for dist in (DIST_UNIFORM, DIST_HOT):
        base = Scenario(distribution=dist, data_gib=10, write_rate=50_000,
                        snapshot_seconds=60, huge_pages=False)
        huge = Scenario(distribution=dist, data_gib=10, write_rate=50_000,
                        snapshot_seconds=60, huge_pages=True)
        a = evaluate(base, LINUX_4K, pressure=False)["peak_gib"]
        b = evaluate(huge, LINUX_4K, pressure=False)["peak_gib"]
        assert b >= a - 1e-9


def test_huge_pages_raise_copy_latency():
    base = Scenario(data_gib=10, write_rate=50_000, snapshot_seconds=60)
    huge = Scenario(data_gib=10, write_rate=50_000, snapshot_seconds=60, huge_pages=True)
    la = evaluate(base, LINUX_4K, pressure=False)["latency"]
    lb = evaluate(huge, LINUX_4K, pressure=False)["latency"]
    assert lb["p99_ms"] >= la["p99_ms"]


# ---------------------------------------------------------------------------
# 延迟
# ---------------------------------------------------------------------------
def test_fork_pause_scales_with_page_count():
    """fork 复制页表的停顿随页数增长；30 GiB/4 KiB 页应在几十毫秒量级。"""
    small = Scenario(data_gib=10, snapshot_seconds=60)
    big = Scenario(data_gib=30, snapshot_seconds=60)
    a = evaluate(small, LINUX_4K, pressure=False)["latency"]["fork_pause_ms"]
    b = evaluate(big, LINUX_4K, pressure=False)["latency"]["fork_pause_ms"]
    assert b > a > 0
    assert 20 < b < 80


def test_fork_pause_shows_up_in_p999():
    sc = Scenario(data_gib=30, write_rate=5_000, snapshot_seconds=30, ram_gib=64)
    r = evaluate(sc, LINUX_4K, pressure=False)
    assert r["latency"]["p999_ms"] > r["latency"]["p99_ms"]
    assert r["latency"]["p999_ms"] >= r["latency"]["fork_pause_ms"]


def test_pressure_raises_latency_when_memory_full():
    low = Scenario(data_gib=10, write_rate=50_000, snapshot_seconds=60, ram_gib=64)
    high = Scenario(data_gib=45, write_rate=200_000, snapshot_seconds=600, ram_gib=64)
    a = evaluate(low, LINUX_4K, pressure=False)["latency"]["p99_ms"]
    b = evaluate(high, LINUX_4K, pressure=False)["latency"]["p99_ms"]
    assert b > a


def test_no_pressure_below_watermark():
    sc = Scenario(data_gib=10, write_rate=10_000, snapshot_seconds=30, ram_gib=64)
    r = evaluate(sc, LINUX_4K, pressure=False)
    assert r["latency"]["pressure_added_p99_ms"] == pytest.approx(0.0, abs=1e-9)
