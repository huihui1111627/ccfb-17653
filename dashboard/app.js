/* 纯原生 canvas 看板，无外部依赖；数据来自 report.js 的 window.REPORT */
const R = window.REPORT;
const GIB = 1024 ** 3;
const $ = id => document.getElementById(id);
const fmt = (b, d = 1) => (b / GIB).toFixed(d) + " GiB";
const WL_CN = { uniform: "均匀随机", hotkey: "热键反复", sequential: "顺序导入", mixed: "大键+小键" };

function setupCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = +cv.height;
  cv.width = w * dpr; cv.height = h * dpr;
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  return { ctx, w, h };
}
function axes(ctx, w, h, pad) {
  ctx.strokeStyle = "#30363d"; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, h - pad.b); ctx.lineTo(w - pad.r, h - pad.b);
  ctx.stroke();
}
function text(ctx, s, x, y, color = "#8b949e", align = "left") {
  ctx.fillStyle = color; ctx.font = "11px monospace"; ctx.textAlign = align;
  ctx.fillText(s, x, y);
}

/* ---------------- KPI ---------------- */
(function kpis() {
  const runs = R.incident.runs;
  const thpOn = runs.find(r => r.huge_on && r.workload === "uniform");
  const thpOff = runs.find(r => !r.huge_on && r.workload === "uniform");
  const val = R.validation;
  const cards = [
    { l: "THP开 · 均匀写峰值内存", v: fmt(thpOn.footprint_peak_bytes), cls: thpOn.oom ? "bad" : "ok" },
    { l: "THP关 · 同场景峰值内存", v: fmt(thpOff.footprint_peak_bytes), cls: "ok" },
    { l: "THP 开/关 复制量放大", v: (thpOn.copied_bytes_peak / thpOff.copied_bytes_peak).toFixed(1) + " ×", cls: "warn" },
    { l: "THP开 真正撞缺页的请求占比", v: thpOn.fault_request_pct.toFixed(1) + " %", cls: "warn" },
    { l: "真实 fork 实测页数误差", v: val.mean_page_count_error_pct.toFixed(2) + " %", cls: "ok" },
    { l: "实测缺页写/普通写延迟", v: (val.rows.reduce((s, r) => s + r.fault_vs_repeat_x, 0) / val.rows.length).toFixed(1) + " ×", cls: "warn" },
  ];
  $("kpis").innerHTML = cards.map(c =>
    `<div class="card kpi"><span class="v ${c.cls}">${c.v}</span><span class="l">${c.l}</span></div>`
  ).join("");
})();

/* ---------------- ① 曲线 ---------------- */
const curveKeys = Object.keys(R.sweep_curves);
function uniq(arr) { return [...new Set(arr)]; }
function fillSelect(el, vals, labels) {
  el.innerHTML = vals.map((v, i) => `<option value="${v}">${labels ? labels[i] : v}</option>`).join("");
}
const parts = curveKeys.map(k => k.split("|"));
fillSelect($("cData"), uniq(parts.map(p => p[1])), uniq(parts.map(p => p[1])));
fillSelect($("cRate"), uniq(parts.map(p => p[2])), uniq(parts.map(p => p[2])));
fillSelect($("cWl"), uniq(parts.map(p => p[0])), uniq(parts.map(p => WL_CN[p[0]] || p[0])));
fillSelect($("cThp"), ["thp0", "thp1"], ["THP关", "THP开"]);

function drawCurve() {
  const key = curveKeys.find(k =>
    k.includes($("cData").value) && k.includes($("cRate").value) &&
    k.startsWith($("cWl").value + "|") && k.endsWith($("cThp").value));
  const cv = $("curve"); const { ctx, w, h } = setupCanvas(cv);
  const pad = { l: 56, r: 56, t: 16, b: 34 };
  ctx.clearRect(0, 0, w, h);
  if (!key) { text(ctx, "该组合无曲线（请换选择）", w / 2, h / 2, "#f85149", "center"); return; }
  const c = R.sweep_curves[key];
  const ram = R.incident.params.ram_gib;
  const maxY = Math.max(ram, ...c.footprint_gib) * 1.05;
  const X = i => pad.l + (c.t[i] / c.t[c.t.length - 1]) * (w - pad.l - pad.r);
  const Y = v => h - pad.b - (v / maxY) * (h - pad.t - pad.b);
  axes(ctx, w, h, pad);
  for (let g = 0; g <= 4; g++) {
    const v = maxY * g / 4, y = Y(v);
    text(ctx, v.toFixed(0), pad.l - 8, y + 3, "#8b949e", "right");
    ctx.strokeStyle = "#21262d"; ctx.beginPath();
    ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
  }
  text(ctx, "GiB", pad.l - 40, pad.t - 4);
  text(ctx, "秒 →", w - pad.r, h - 10, "#8b949e", "right");
  // RAM 阈值
  ctx.strokeStyle = "#8b949e"; ctx.setLineDash([6, 5]);
  ctx.beginPath(); ctx.moveTo(pad.l, Y(ram)); ctx.lineTo(w - pad.r, Y(ram)); ctx.stroke();
  ctx.setLineDash([]);
  text(ctx, `物理内存 ${ram}GiB`, w - pad.r, Y(ram) - 5, "#8b949e", "right");
  // footprint
  drawLine(ctx, c.t.map((_, i) => [X(i), Y(c.footprint_gib[i])]), "#388bfd", 2);
  // copied
  drawLine(ctx, c.t.map((_, i) => [X(i), Y(c.copied_gib[i])]), "#f85149", 2);
  // OOM 标记
  if (c.oom_at != null) {
    const oi = c.t.findIndex(t => t >= c.oom_at);
    if (oi >= 0) {
      const x = X(oi);
      ctx.strokeStyle = "#f85149"; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
      ctx.setLineDash([]);
      text(ctx, "OOM 杀进程", x, pad.t + 10, "#f85149", "center");
    }
  }
  const peak = Math.max(...c.footprint_gib);
  text(ctx, `峰值 ${peak.toFixed(1)} GiB`, pad.l + 6, pad.t + 12, "#388bfd");
  text(ctx, `复制峰值 ${Math.max(...c.copied_gib).toFixed(2)} GiB`, pad.l + 6, pad.t + 26, "#f85149");
}
function drawLine(ctx, pts, color, lw) {
  ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.beginPath();
  pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
  ctx.stroke();
}
["cData", "cRate", "cWl", "cThp"].forEach(id => $(id).addEventListener("change", drawCurve));
drawCurve();

/* ---------------- ② 热力图 ---------------- */
const HM = R.heatmap;
/* 把滑块的线性进度映射成真实复制曲线上的复制比例（指数衰减形状，而非线性）。
   无曲线数据时退化为线性。 */
function progressToCopyFrac(progress) {
  const curve = HM.copy_frac_curve;
  if (!curve || !curve.length) return progress / 100;
  const pos = (progress / 100) * (curve.length - 1);
  const i = Math.floor(pos), f = pos - i;
  const a = curve[i], b = curve[Math.min(curve.length - 1, i + 1)];
  return a + (b - a) * f;
}
function drawHeat(progress) {
  const cv = $("heatmap"); const { ctx, w, h } = setupCanvas(cv);
  ctx.clearRect(0, 0, w, h);
  const cw = w / HM.width, ch = h / HM.height;
  const alpha = progressToCopyFrac(progress);
  for (let y = 0; y < HM.height; y++) {
    for (let x = 0; x < HM.width; x++) {
      const live = HM.live[y][x], hole = HM.hole[y][x], copy = HM.copy_final[y][x] * alpha;
      let r, g, b;
      // 复制红覆盖在底色之上，按 copy 强度混合
      const baseR = live * 31 + hole * 68, baseG = live * 111 + hole * 76, baseB = live * 235 + hole * 86;
      r = baseR * (1 - copy) + 248 * copy;
      g = baseG * (1 - copy) + 81 * copy;
      b = baseB * (1 - copy) + 73 * copy;
      ctx.fillStyle = `rgb(${r | 0},${g | 0},${b | 0})`;
      ctx.fillRect(x * cw, y * ch, Math.ceil(cw), Math.ceil(ch));
    }
  }
}
drawHeat(100);
let heatTimer = null;
$("heatRange").addEventListener("input", e => {
  $("heatPct").textContent = e.target.value + "%";
  drawHeat(+e.target.value);
});
$("heatPlay").addEventListener("change", e => {
  if (!e.target.checked) { clearInterval(heatTimer); return; }
  heatTimer = setInterval(() => {
    let v = (+$("heatRange").value + 2) % 102;
    $("heatRange").value = v; $("heatPct").textContent = v + "%";
    drawHeat(v);
  }, 120);
});

/* ---------------- ③ 延迟对齐 ---------------- */
function drawLatency(cvId, d) {
  const cv = $(cvId); const { ctx, w, h } = setupCanvas(cv);
  const pad = { l: 50, r: 46, t: 14, b: 30 };
  ctx.clearRect(0, 0, w, h);
  const maxUs = Math.max(...d.max_us) * 1.1;
  const X = i => pad.l + (d.t[i] / d.t[d.t.length - 1]) * (w - pad.l - pad.r);
  const Y = v => h - pad.b - (v / maxUs) * (h - pad.t - pad.b);
  axes(ctx, w, h, pad);
  text(ctx, "µs", pad.l - 34, pad.t - 2);
  for (let g = 0; g <= 3; g++) {
    const v = maxUs * g / 3;
    text(ctx, v.toFixed(0), pad.l - 6, Y(v) + 3, "#8b949e", "right");
  }
  // 故障率柱（背景橙）
  const maxFr = Math.max(...d.fault_rate, 0.01);
  d.fault_rate.forEach((fr, i) => {
    ctx.fillStyle = "rgba(210,153,34,0.35)";
    const bh = (fr / maxFr) * (h - pad.t - pad.b) * 0.4;
    ctx.fillRect(X(i) - 1, h - pad.b - bh, 2, bh);
  });
  drawLine(ctx, d.t.map((_, i) => [X(i), Y(d.p50_us[i])]), "#3fb950", 1.5);
  drawLine(ctx, d.t.map((_, i) => [X(i), Y(d.p99_us[i])]), "#d29922", 1.8);
  drawLine(ctx, d.t.map((_, i) => [X(i), Y(d.max_us[i])]), "#f85149", 1.5);
  text(ctx, "p50", w - pad.r - 90, pad.t + 8, "#3fb950");
  text(ctx, "p99", w - pad.r - 55, pad.t + 8, "#d29922");
  text(ctx, "max", w - pad.r - 20, pad.t + 8, "#f85149", "right");
  text(ctx, "橙柱=缺页故障率（与尖刺对齐）", pad.l + 4, h - 8, "#8b949e");
}
drawLatency("latBase", R.latency_base);
drawLatency("latHuge", R.latency_huge);

/* ---------------- ④ 事故表 ---------------- */
(function incident() {
  const rows = R.incident.runs;
  let html = "<tr><th>大页</th><th>写入模式</th><th>峰值内存</th><th>COW复制</th>" +
    "<th>fork停顿(ms)</th><th>p50(µs)</th><th>p99(µs)</th><th>p999(µs)</th><th>故障请求%</th><th>结果</th></tr>";
  rows.forEach(r => {
    html += `<tr>
      <td><span class="tag ${r.huge_on ? "on" : "off"}">${r.huge_on ? "THP开" : "THP关"}</span></td>
      <td style="text-align:left">${WL_CN[r.workload] || r.workload}</td>
      <td class="${r.oom ? "bad" : ""}">${fmt(r.footprint_peak_bytes)}</td>
      <td>${fmt(r.copied_bytes_peak, 2)}</td>
      <td>${r.fork_stall_ms.toFixed(1)}</td>
      <td>${r.p50_us.toFixed(0)}</td>
      <td class="${r.p99_us > 3000 ? "bad" : ""}">${r.p99_us.toFixed(0)}</td>
      <td>${r.p999_us.toFixed(0)}</td>
      <td>${r.fault_request_pct.toFixed(1)}</td>
      <td class="${r.oom ? "bad" : "ok"}">${r.oom ? "OOM 被杀" : "存活"}</td></tr>`;
  });
  $("incidentTable").innerHTML = html;
})();

/* ---------------- ⑤ 网格表 ---------------- */
(function sweep() {
  const wls = uniq(R.sweep_rows.map(r => r.workload));
  wls.forEach(w => $("fWl").innerHTML += `<option value="${w}">${WL_CN[w] || w}</option>`);
  function render() {
    const fT = $("fThp").value, fW = $("fWl").value;
    const rows = R.sweep_rows.filter(r =>
      (fT === "" || (r.huge_on ? "1" : "0") === fT) && (fW === "" || r.workload === fW));
    let html = "<tr><th>数据</th><th>大页</th><th>模式</th><th>速率/s</th><th>快照s</th>" +
      "<th>峰值</th><th>复制GiB</th><th>p99µs</th><th>p999µs</th><th>公式误差%</th><th>结果</th></tr>";
    rows.forEach(r => {
      html += `<tr>
        <td>${r.data_gib}G</td>
        <td>${r.huge_on ? "开" : "关"}</td>
        <td style="text-align:left">${WL_CN[r.workload] || r.workload}</td>
        <td>${r.rate}</td><td>${r.snapshot_s}</td>
        <td class="${r.oom ? "bad" : ""}">${(r.footprint_peak_bytes / GIB).toFixed(1)}</td>
        <td>${(r.copied_bytes_peak / GIB).toFixed(2)}</td>
        <td class="${r.p99_us > 3000 ? "bad" : ""}">${r.p99_us.toFixed(0)}</td>
        <td>${r.p999_us.toFixed(0)}</td>
        <td>${r.formula_sim_err_pct.toFixed(1)}</td>
        <td class="${r.oom ? "bad" : "ok"}">${r.oom ? "OOM" : "OK"}</td></tr>`;
    });
    $("sweepTable").innerHTML = html;
  }
  $("fThp").addEventListener("change", render);
  $("fWl").addEventListener("change", render);
  render();
})();

/* ---------------- ⑥ 实测表 ---------------- */
(function valid() {
  const v = R.validation;
  let html = "<tr><th>模式</th><th>页大小</th><th>实测触碰页</th><th>内核故障页</th>" +
    "<th>仿真页数</th><th>页数误差</th><th>故障写p50µs</th><th>重复写p50µs</th><th>延迟放大</th></tr>";
  v.rows.forEach(r => {
    html += `<tr>
      <td style="text-align:left">${WL_CN[r.mode] || r.mode}</td>
      <td>${(r.page_size / 1024) + "KiB"}</td>
      <td>${r.real_unique_pages}</td><td>${r.kernel_cow_faults}</td>
      <td>${r.sim_unique_pages}</td>
      <td class="ok">${r.page_count_error_pct.toFixed(2)}%</td>
      <td class="warn">${r.real_fault_p50_us.toFixed(2)}</td>
      <td>${r.real_repeat_p50_us.toFixed(2)}</td>
      <td class="warn">${r.fault_vs_repeat_x.toFixed(1)}×</td></tr>`;
  });
  $("validTable").innerHTML = html;
  $("validNote").textContent =
    `平台 ${v.platform}，竞技场 ${v.arena_mib} MiB，${v.writes} 次写。` +
    `复制页数以 getrusage(RUSAGE_SELF).ru_minflt 次缺页差值为内核精确计数；` +
    `仿真与实测页号序列完全对齐故页数误差为 0。${v.calibrated_model.note}。` +
    `macOS 内存压缩使 footprint 差值失真（约 100% 偏小），因此不采用该口径。`;
})();

/* ---------------- ⑦ 建议 ---------------- */
(function recs() {
  $("recs").innerHTML = R.recommend.recommendations.map(r =>
    `<div class="rec"><h3>${r.title}</h3><p>${r.detail}</p></div>`).join("");
  let html = "<tr><th>场景</th><th>峰值内存</th><th>COW复制</th><th>p99µs</th>" +
    "<th>p999µs</th><th>故障请求%</th><th>结果</th></tr>";
  Object.entries(R.recommend.scenarios).forEach(([name, s]) => {
    html += `<tr>
      <td style="text-align:left">${name}</td>
      <td class="${s.oom ? "bad" : "ok"}">${fmt(s.footprint_peak_bytes)}</td>
      <td>${fmt(s.copied_bytes_peak, 2)}</td>
      <td>${s.p99_us.toFixed(0)}</td><td>${s.p999_us.toFixed(0)}</td>
      <td>${s.fault_request_pct.toFixed(1)}</td>
      <td class="${s.oom ? "bad" : "ok"}">${s.oom ? "OOM" : "存活"}</td></tr>`;
  });
  $("scenTable").innerHTML = html;
})();
