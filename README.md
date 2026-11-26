# fork 快照的 COW 峰值内存与请求延迟

64 GiB 的缓存机器、30 GB 数据，每天持久化那几分钟内存涨到近 60 GiB、被 OOM 杀两次、
延迟跳到几十毫秒。这个项目用 **页级模拟 + 真机 fork 实测** 把这件事拆开算清楚，
并给出可执行的调整建议。

**先读 [`reports/report.md`](reports/report.md)**，那里有完整分析。
交互页面在 [`web/index.html`](web/index.html)。

---

## 一句话结论

峰值内存 = 数据常驻 + 快照窗口内被写过的页 × 页大小，**硬上界是 2 倍**。
决定它的是 `写入速率 × 快照时长 / 页数`——"写太多"和"快照太慢"是同一个旋钮。

## 快速开始

```bash
# 交互页面（无依赖，直接起静态服务）
cd web && python3 -m http.server 8765
# 打开 http://localhost:8765

# 跑测试
python3 -m pytest tests -q        # 48 passed

# 重建全部产物（标定 → 实测 → 报告 → 页面数据 → 静态图，约 1 分钟）
./rebuild.sh
```

## 目录

```
sim/          模型与仿真
  profiles.py   平台常数：页大小、COW 时延、fork 停顿（实测/文献标注）
  layout.py     键值到页：slab 打包、大键值跨页、删除留洞、脏页计算
  heat.py       块热度分组（均匀/热键/Zipf/顺序），解析与仿真共用
  model.py      COW 复制曲线、峰值内存、内存压力、延迟模型
  workload.py   写入轨迹生成器
  pagesim.py    页级仿真（逐块位图 / 桶近似）
bench/        真机 fork 实测
  fork_bench.py   真正的 fork + COW，macOS 私有 API 精确数页
  run_matrix.py   实测矩阵
analysis/     汇总与产物
  build_report.py   实测 vs 推演、场景矩阵、建议效果
  build_web_data.py 页面数据
  make_figures.py   报告静态图
calib/        本机常数标定
web/          交互页面（单文件，无依赖）
reports/      长文报告 + summary.json
tests/        48 个模型不变量测试
figures/      报告用静态图
```

## 这个模型能回答什么

- 30 GiB 数据在 64 GiB 机器上，什么写入速率 / 快照时长组合会 OOM；
- 均匀写、顺序写、热键、Zipf 长尾四种业务形态的峰值内存差别有多大；
- 开 2 MiB 大页后，单次 COW 停顿、COW CPU、峰值内存各自恶化多少；
- 延迟的几十毫秒到底来自 COW 拷贝、内存回收，还是 fork 本身；
- 单机放多少数据安全；
- 关大页 / 缩短快照 / 数据重排 / 压数据量，各自能救回多少。

## 模型不覆盖什么

私有匿名页之外的（文件页、WAL）、快照子进程自身的内存写入、内核回收器的精确行为。
详见报告第九节。

## 实测环境

macOS 26.7 / arm64 / 16 KiB 页 / Python 3.9 / NumPy 2.0。
Linux 4 KiB 与 THP 的常数来自公开资料，标注在 `sim/profiles.py` 里。
