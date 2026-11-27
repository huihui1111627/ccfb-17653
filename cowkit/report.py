"""汇总所有推演结果，生成 dashboard 使用的单个 report JSON + report.js。

包含
====
* incident    : 30/64GiB 事故复现（THP 开关 × 四模式），含时间曲线与热力图
* sweep       : 192 组参数网格的汇总指标
* validation  : 真实 fork 实测 vs 仿真的误差表
* recommend   : 五条建议与调整后场景推演
* latency     : 一条带“缺页事件”对齐标记的请求延迟时间序列（dashboard 对齐视图）
"""

from __future__ import annotations

import json
import os

import numpy as np

from .pages import ArenaConfig, PageArena, THP_2M
from .workloads import WorkloadSpec
from .engine import simulate_snapshot, LatencyModel
from .sweep import reproduce_incident

GIB = 1024 ** 3


def build_latency_alignment(huge: bool = False) -> dict:
    """造一条 60s 快照片段的延迟序列，并标出每个请求是否撞上 COW 缺页，
    供 dashboard 把延迟尖刺和页复制事件在同一时间轴上对齐。

    竞技场 16 GiB、速率 600 写/秒：大页容器约 8 千个，故障风暴在十几秒内衰减；
    基础页容器约 4 百万个，整窗几乎每写必故障但每次仅 8us，曲线保持平。
    两者用相同参数，凸显“同样的写、复制单位不同 => 延迟形态完全不同”。"""
    cfg = ArenaConfig(data_bytes=16 * GIB, page_size=4096,
                      huge_size=(THP_2M if huge else 0),
                      fill_fraction=0.85, seed=51)
    arena = PageArena(cfg)
    spec = WorkloadSpec("uniform", write_rate=600, duration=60, seed=51)
    # 低基础延迟模型（中位 100us）：让 150us 大页故障形成可见尖刺，
    # 而 8us 基础页故障几乎不可见，凸显复制单位对尾延迟的影响差异。
    align_model = LatencyModel(base_median_us=100.0, base_p99_us=260.0,
                               rng_seed=51)
    res = simulate_snapshot(arena, spec, ram_bytes=64 * GIB,
                            model=align_model, n_buckets=60, heatmap_cells=600)
    return {
        "t": res.t.tolist(),
        "p50_us": _bucket_quantile(res, 50),
        "p99_us": _bucket_quantile(res, 99),
        "max_us": _bucket_quantile(res, 100),
        "copied_pages": res.copied_pages.tolist(),
        "fault_rate": res.fault_req_rate.tolist(),
        "huge_on": huge,
    }


def _bucket_quantile(res, q):
    # lat_all 是全量样本，按桶等长切分求分位
    n = len(res.t)
    if res.lat_all_us.size == 0:
        return [0.0] * n
    chunks = np.array_split(res.lat_all_us, n)
    if q == 100:
        return [float(c.max()) if c.size else 0.0 for c in chunks]
    return [float(np.percentile(c, q)) if c.size else 0.0 for c in chunks]


def build_report(results_dir: str = "results") -> dict:
    incident = json.load(open(os.path.join(results_dir, "incident.json")))
    sweep = json.load(open(os.path.join(results_dir, "sweep.json")))
    validation = json.load(open(os.path.join(results_dir, "validation.json")))
    recommend = json.load(open(os.path.join(results_dir, "recommend.json")))

    # 事故复现补一份热力图（uniform/THP开，最有代表性的 OOM 场景）
    cfg = ArenaConfig(data_bytes=30 * GIB, page_size=4096, huge_size=THP_2M,
                      fill_fraction=0.85, seed=21)
    arena = PageArena(cfg)
    spec = WorkloadSpec("uniform", 4000, 180, seed=21)
    heat_res = simulate_snapshot(arena, spec, ram_bytes=64 * GIB,
                                 n_buckets=120, heatmap_cells=2400)
    heatmap = heat_res.heatmap
    heatmap["footprint_curve_gib"] = (heat_res.footprint_bytes / GIB).tolist()
    heatmap["copied_curve_gib"] = (heat_res.copied_pages * 4096 / GIB).tolist()
    heatmap["t"] = heat_res.t.tolist()

    report = {
        "title": "Redis 类键值库 fork()/COW 内存与延迟仿真报告",
        "incident": incident,
        "sweep_meta": sweep["meta"],
        "sweep_rows": sweep["rows"],
        "sweep_curves": sweep["curves"],
        "validation": validation,
        "recommend": recommend,
        "heatmap": heatmap,
        "latency_base": build_latency_alignment(huge=False),
        "latency_huge": build_latency_alignment(huge=True),
    }
    return report


def write_report(results_dir: str = "results",
                 dashboard_dir: str = "dashboard") -> str:
    report = build_report(results_dir)
    json_path = os.path.join(dashboard_dir, "report.json")
    js_path = os.path.join(dashboard_dir, "report.js")
    os.makedirs(dashboard_dir, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(report, f, ensure_ascii=False)
    with open(js_path, "w") as f:
        f.write("window.REPORT = ")
        json.dump(report, f, ensure_ascii=False)
        f.write(";\n")
    size_mib = os.path.getsize(js_path) / 1024 ** 2
    print(f"report -> {json_path} / {js_path} ({size_mib:.2f} MiB)")
    return js_path


if __name__ == "__main__":
    write_report()
