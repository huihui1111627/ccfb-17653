#!/usr/bin/env bash
# 一键重建：标定 → 实测 → 报告 → 页面数据 → 静态图
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1/5 本机标定 =="
python3 calib/calibrate.py > /dev/null
echo "   calib/measured.json"

echo "== 2/5 真机 fork 实测矩阵（约 1 分钟）=="
python3 bench/run_matrix.py all | tail -4

echo "== 3/5 模型 + 报告汇总 =="
python3 analysis/build_report.py | head -6

echo "== 4/5 页面数据 =="
python3 analysis/build_web_data.py

echo "== 5/5 静态图 =="
python3 analysis/make_figures.py 2>/dev/null

echo
echo "完成。打开页面： cd web && python3 -m http.server 8765"
echo "长文报告：   reports/report.md"
