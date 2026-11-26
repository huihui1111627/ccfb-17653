"""
生成报告用的静态图（matplotlib）。

图：
  figures/incident.png   现场推演曲线
  figures/latency.png    延迟对齐
  figures/modes.png      四种写入模式并排
  figures/huge.png       大页 vs 基页
  figures/bench.png      实测 vs 推演
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from sim.model import (DIST_HOT, DIST_SEQUENTIAL, DIST_UNIFORM, DIST_ZIPF,
                       Scenario, simulate)
from sim.pagesim import BucketSim
from sim.profiles import LINUX_4K

FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "PingFang HK",
                                 "Heiti TC", "Songti SC", "Arial Unicode MS",
                                 "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "figure.facecolor": "#0e1116", "axes.facecolor": "#0e1116",
    "savefig.facecolor": "#0e1116", "text.color": "#dbe3ee",
    "axes.labelcolor": "#8b97a8", "xtick.color": "#8b97a8",
    "ytick.color": "#8b97a8", "axes.edgecolor": "#2a3240",
    "grid.color": "#20283a", "font.size": 10,
})
GIB = 1 << 30


def incident():
    sc = Scenario(name="incident", data_gib=30, util=0.85, write_rate=50_000,
                  distribution=DIST_UNIFORM, locality=0.3, snapshot_seconds=600,
                  ram_gib=64, ncpu=8, huge_pages=False)
    tr = BucketSim(sc, seed=1, n_samples=200).run()
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=160)
    t = tr.t
    ax.plot(t, tr.memory_bytes / GIB, color="#5aa9ff", lw=2, label="父进程数据区内存")
    ax.fill_between(t, tr.resident_bytes / GIB, tr.memory_bytes / GIB,
                    color="#5aa9ff", alpha=0.14)
    ax.axhline(64, color="#ff6b6b", ls="--", lw=1.2, label="64 GiB 物理内存")
    ax.axhline(64 * 0.85, color="#ffb454", ls=":", lw=1.2, label="85% 回收水线")
    ax.axhline(tr.resident_bytes / GIB, color="#8b97a8", ls=":", lw=1,
               label="数据常驻")
    ax.set_xlabel("快照进行时间 (秒)")
    ax.set_ylabel("内存 (GiB)")
    ax.set_title("现场推演：30 GiB 数据 · 50k 写/秒 · 600 s 慢快照 · 关 THP",
                 color="#dbe3ee", fontsize=11)
    ax.legend(loc="lower right", framealpha=0.1, labelcolor="#dbe3ee", fontsize=8.5)
    ax.grid(True, alpha=0.35)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(30, 74)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "incident.png"))
    plt.close(fig)


def latency():
    sc = Scenario(name="incident", data_gib=30, util=0.85, write_rate=50_000,
                  distribution=DIST_UNIFORM, locality=0.3, snapshot_seconds=600,
                  ram_gib=64, ncpu=8, huge_pages=False)
    tr = BucketSim(sc, seed=1, n_samples=200).run()
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=160)
    cow = tr.lat_cow
    ax.scatter(tr.lat_t[~cow], tr.lat_ms[~cow], s=7, color="#8b97a8",
               alpha=0.5, label="普通请求")
    ax.scatter(tr.lat_t[cow], tr.lat_ms[cow], s=7, color="#ff6b6b",
               alpha=0.7, label="命中 COW")
    ax.axhline(8, color="#ffb454", ls="--", lw=1, label="基线 p99 ≈ 8 ms")
    ax.set_xlabel("快照进行时间 (秒)")
    ax.set_ylabel("请求延迟 (ms)")
    ax.set_title("请求延迟与 COW 事件对齐：前段 COW 密集但很快，后段压力停顿主导",
                 color="#dbe3ee", fontsize=11)
    ax.legend(loc="upper left", framealpha=0.1, labelcolor="#dbe3ee", fontsize=8.5)
    ax.grid(True, alpha=0.35)

    # 放大前 160 s：展示 COW 本身只让延迟从 1.6 抬到 1.61~1.62 ms
    ins = ax.inset_axes([0.52, 0.12, 0.44, 0.34])
    m = tr.lat_t <= 160
    ins.scatter(tr.lat_t[m & ~cow], tr.lat_ms[m & ~cow], s=5, color="#8b97a8", alpha=0.5)
    ins.scatter(tr.lat_t[m & cow], tr.lat_ms[m & cow], s=5, color="#ff6b6b", alpha=0.7)
    ins.set_ylim(1.5, 1.75)
    ins.set_title("放大前 160 s：COW 只加了几微秒", color="#8b97a8", fontsize=7.5)
    ins.tick_params(labelsize=6.5)
    ins.set_facecolor("#131821")
    ins.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "latency.png"))
    plt.close(fig)


def modes():
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), dpi=150, sharey=True)
    specs = [(DIST_UNIFORM, "均匀随机写"), (DIST_SEQUENTIAL, "顺序写（批量导入）"),
             (DIST_HOT, "少数热键反复写"), (DIST_ZIPF, "Zipf 长尾写")]
    for ax, (dist, label) in zip(axes.flat, specs):
        sc = Scenario(name=dist, data_gib=30, util=0.85, write_rate=50_000,
                      distribution=dist, locality=1.0, snapshot_seconds=120,
                      ram_gib=64, ncpu=8)
        tr = BucketSim(sc, seed=2, n_samples=160).run()
        ax.plot(tr.t, tr.memory_bytes / GIB, color="#5aa9ff", lw=1.8)
        ax.axhline(tr.resident_bytes / GIB, color="#8b97a8", ls=":", lw=1)
        peak = tr.memory_bytes.max() / GIB
        ratio = tr.memory_bytes.max() / tr.resident_bytes
        ax.set_title(f"{label}  ·  {peak:.1f} GiB ({ratio:.2f}×)",
                     color="#dbe3ee", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, tr.t[-1])
    for ax in axes[1]:
        ax.set_xlabel("时间 (秒)", color="#8b97a8")
    for ax in axes[:, 0]:
        ax.set_ylabel("内存 (GiB)", color="#8b97a8")
    fig.suptitle("同样 30 GiB / 50k 写/秒 / 120 s，只是写入分布不同",
                 color="#dbe3ee", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "modes.png"))
    plt.close(fig)


def huge():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    for ax, huge_on, color, label in ((ax1, False, "#5aa9ff", "基页"),
                                      (ax1, True, "#c792ea", "2 MiB 大页")):
        sc = Scenario(name="h", data_gib=30, util=0.85, write_rate=50_000,
                      distribution=DIST_UNIFORM, locality=1.0, snapshot_seconds=120,
                      ram_gib=64, ncpu=8, huge_pages=huge_on)
        tr = BucketSim(sc, seed=3, n_samples=160).run()
        ax.plot(tr.t, tr.memory_bytes / GIB, color=color, lw=1.9, label=label)
    ax1.axhline(64, color="#ff6b6b", ls="--", lw=1)
    ax1.set_title("开大页后，很小的写入量就把数据复制光", color="#dbe3ee", fontsize=10)
    ax1.set_xlabel("时间 (秒)"); ax1.set_ylabel("内存 (GiB)")
    ax1.legend(framealpha=0.1, labelcolor="#dbe3ee", fontsize=8.5)
    ax1.grid(True, alpha=0.3)

    # 单次停顿柱状图（来自实测）
    labels = ["16 KiB 基页\np50", "16 KiB 基页\np99", "2 MiB 大页\np50", "2 MiB 大页\np99"]
    vals = [2.5, 14.6, 236, 278]
    colors = ["#5aa9ff", "#5aa9ff", "#c792ea", "#c792ea"]
    ax2.bar(range(4), vals, color=colors, alpha=0.85)
    ax2.set_yscale("log")
    ax2.set_xticks(range(4)); ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("单次 COW 停顿 (µs，对数)")
    ax2.set_title("实测：写一页触发的停顿时延", color="#dbe3ee", fontsize=10)
    for i, v in enumerate(vals):
        ax2.text(i, v * 1.15, f"{v:g}", ha="center", color="#dbe3ee", fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "huge.png"))
    plt.close(fig)


def bench():
    summary_path = os.path.join(ROOT, "reports", "summary.json")
    if not os.path.exists(summary_path):
        return
    rows = [r for r in json.load(open(summary_path))["bench_vs_model"]
            if r["group"] == "memory"]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4), dpi=150)
    labels = [r["name"].replace("mem_", "") for r in rows]
    meas = [r["measured_peak_ratio"] for r in rows]
    model = [r["model_peak_ratio"] for r in rows]
    x = np.arange(len(rows))
    w = 0.38
    ax.bar(x - w/2, meas, w, label="实测", color="#57d38c", alpha=0.9)
    ax.bar(x + w/2, model, w, label="推演", color="#5aa9ff", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=8)
    ax.set_ylabel("峰值倍数")
    ax.axhline(2.0, color="#ff6b6b", ls="--", lw=1, alpha=0.7)
    ax.set_title("真机 fork 实测 vs 模型推演", color="#dbe3ee", fontsize=11)
    ax.legend(framealpha=0.1, labelcolor="#dbe3ee", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "bench.png"))
    plt.close(fig)


def main():
    incident()
    latency()
    modes()
    huge()
    bench()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
