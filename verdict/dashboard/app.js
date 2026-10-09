"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const fmt = {
  int: (n) => (n == null ? "–" : Math.round(n).toLocaleString()),
  ms: (n) => (n == null ? "–" : n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1) + " s" : Math.round(n) + " ms"),
  pct: (n, d = 0) => (n == null ? "–" : (n * 100).toFixed(d) + "%"),
  num: (n, d = 3) => (n == null ? "–" : Number(n).toFixed(d)),
  time: (t) => new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
  datetime: (t) => new Date(t * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" }),
};

const state = { range: 3600, models: [], backends: {}, timer: null };

function getToken() {
  try { return localStorage.getItem("verdict.token") || ""; } catch { return ""; }
}

async function api(path, opts = {}, retried = false) {
  const headers = { "Content-Type": "application/json" };
  const token = getToken();
  if (token) headers.Authorization = "Bearer " + token;
  const r = await fetch(path, { ...opts, headers, body: opts.body ? JSON.stringify(opts.body) : undefined });
  // Server started with --token: ask once, remember it in this browser, retry.
  if (r.status === 401 && !retried) {
    const t = prompt("This Verdict server needs its access token (the --token it was started with):");
    if (t) {
      try { localStorage.setItem("verdict.token", t.trim()); } catch {}
      return api(path, opts, true);
    }
  }
  const text = await r.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { detail: text.slice(0, 200) }; }
  if (!r.ok) throw new Error((data && (data.detail?.[0]?.msg || data.detail)) || r.statusText);
  return data;
}

function toast(msg, error = false) {
  const el = document.createElement("div");
  el.className = "toast" + (error ? " error" : "");
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), error ? 7000 : 3500);
}

// Colour follows the model (registration order), never its rank in a view.
function colorOf(name) {
  const i = state.models.findIndex((m) => m.name === name);
  const slot = i < 0 ? 8 : (i % 8) + 1;
  return `var(--series-${slot})`;
}

// ---------------------------------------------------------------- tabs & theme
$$("nav button").forEach((b) =>
  b.addEventListener("click", () => {
    $$("nav button").forEach((x) => x.setAttribute("aria-selected", x === b));
    $$("main > section").forEach((s) => (s.hidden = s.id !== "tab-" + b.dataset.tab));
    try { localStorage.setItem("verdict.tab", b.dataset.tab); } catch {}
    if (b.dataset.tab === "overview") refreshOverview();
  })
);
$$("#range button").forEach((b) =>
  b.addEventListener("click", () => {
    $$("#range button").forEach((x) => x.setAttribute("aria-pressed", x === b));
    state.range = +b.dataset.range;
    refreshOverview();
  })
);
$("#theme").addEventListener("click", () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("verdict.theme", root.dataset.theme); } catch {}
  refreshOverview();
});
try {
  const t = localStorage.getItem("verdict.theme");
  if (t) document.documentElement.dataset.theme = t;
} catch {}

// ---------------------------------------------------------------- line chart
// Axis max whose quarter is a round step (1, 2 or 5 × 10^n) so all 4 ticks are clean numbers.
function niceMax(v) {
  if (!v || v <= 0) return 4;
  const raw = v / 4, p = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * p).find((m) => m >= raw);
  return step * 4;
}

function lineChart(el, { series, since, until, bucket, field, format, fillZero }) {
  const names = Object.keys(series);
  if (!names.length) {
    el.innerHTML = `<div class="empty">No decisions in this range yet.</div>`;
    return;
  }
  const W = Math.max(el.clientWidth, 280), H = 220, m = { l: 48, r: 12, t: 8, b: 26 };
  const n = Math.max(1, Math.round((until - since) / bucket));
  const centers = Array.from({ length: n }, (_, i) => since + (i + 0.5) * bucket);
  const data = {};
  let maxV = 0;
  for (const name of names) {
    const byB = new Map(series[name].map((p) => [Math.floor((p.t - since) / bucket), p[field]]));
    data[name] = centers.map((_, i) => (byB.has(i) ? byB.get(i) : fillZero ? 0 : null));
    for (const v of data[name]) if (v != null) maxV = Math.max(maxV, v);
  }
  const yMax = niceMax(maxV);
  const x = (t) => m.l + ((t - since) / (until - since)) * (W - m.l - m.r);
  const y = (v) => H - m.b - (v / yMax) * (H - m.t - m.b);

  let svg = `<svg viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="line chart">`;
  for (let k = 0; k <= 4; k++) {
    const v = (yMax * k) / 4, yy = y(v);
    svg += `<line class="${k ? "gridline" : "baseline"}" x1="${m.l}" x2="${W - m.r}" y1="${yy}" y2="${yy}"/>`;
    svg += `<text class="tick" x="${m.l - 8}" y="${yy + 4}" text-anchor="end">${format(v)}</text>`;
  }
  const ticks = Math.min(6, Math.floor((W - m.l - m.r) / 90));
  for (let k = 0; k <= ticks; k++) {
    const t = since + ((until - since) * k) / ticks;
    svg += `<text class="tick" x="${x(t)}" y="${H - 6}" text-anchor="${k === 0 ? "start" : k === ticks ? "end" : "middle"}">${fmt.time(t)}</text>`;
  }
  for (const name of names) {
    let d = "", pen = false;
    data[name].forEach((v, i) => {
      if (v == null) { pen = false; return; }
      d += `${pen ? "L" : "M"}${x(centers[i]).toFixed(1)},${y(v).toFixed(1)}`;
      pen = true;
    });
    // isolated points need a visible mark
    const lone = data[name].map((v, i) => (v != null && data[name][i - 1] == null && data[name][i + 1] == null ? i : -1)).filter((i) => i >= 0);
    svg += `<path class="line" d="${d}" stroke="${colorOf(name)}"/>`;
    for (const i of lone) svg += `<circle cx="${x(centers[i])}" cy="${y(data[name][i])}" r="3" fill="${colorOf(name)}"/>`;
  }
  svg += `<g class="hover" style="display:none"><line class="cross" y1="${m.t}" y2="${H - m.b}"/></g>`;
  svg += `<rect x="${m.l}" y="0" width="${W - m.l - m.r}" height="${H}" fill="transparent" class="hit"/></svg>`;

  const legend = names.length > 1 ? `<div class="legend">${names.map((nm) => `<span><span class="swatch" style="background:${colorOf(nm)}"></span>${esc(nm)}</span>`).join("")}</div>` : "";
  el.innerHTML = legend + svg + `<div class="tooltip" hidden></div>`;

  const svgEl = $("svg", el), hover = $(".hover", el), tip = $(".tooltip", el), cross = $(".cross", el);
  const move = (ev) => {
    const rect = svgEl.getBoundingClientRect();
    const px = ((ev.clientX - rect.left) / rect.width) * W;
    const i = Math.max(0, Math.min(n - 1, Math.floor(((px - m.l) / (W - m.l - m.r)) * n)));
    const cx = x(centers[i]);
    hover.style.display = "";
    cross.setAttribute("x1", cx); cross.setAttribute("x2", cx);
    $$(".dot", hover).forEach((d) => d.remove());
    let rows = "";
    for (const name of names) {
      const v = data[name][i];
      if (v == null) continue;
      const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      c.setAttribute("class", "dot"); c.setAttribute("cx", cx); c.setAttribute("cy", y(v)); c.setAttribute("r", 4.5);
      c.setAttribute("fill", colorOf(name));
      hover.appendChild(c);
      rows += `<div class="r"><span><span class="swatch" style="background:${colorOf(name)}"></span>${esc(name)}</span><span>${format(v)}</span></div>`;
    }
    tip.innerHTML = `<div class="t">${fmt.time(centers[i] - bucket / 2)} – ${fmt.time(centers[i] + bucket / 2)}</div>${rows || '<div class="muted">no data</div>'}`;
    tip.hidden = false;
    const scale = rect.width / W, left = cx * scale + (legend ? 0 : 0);
    const tw = tip.offsetWidth;
    tip.style.left = (left + 14 + tw > rect.width ? left - tw - 14 : left + 14) + "px";
    tip.style.top = (svgEl.offsetTop + 10) + "px";
  };
  $(".hit", el).addEventListener("mousemove", move);
  $(".hit", el).addEventListener("mouseleave", () => { hover.style.display = "none"; tip.hidden = true; });
}

// ---------------------------------------------------------------- overview
async function refreshOverview() {
  if ($("#tab-overview").hidden) return;
  const buckets = 48;
  let stats, recent;
  try {
    [stats, recent] = await Promise.all([
      api(`/api/stats?range=${state.range}&buckets=${buckets}`),
      api(`/api/decisions?limit=25&since=${Date.now() / 1000 - state.range}`),
    ]);
  } catch (e) {
    toast("Could not load stats: " + e.message, true);
    return;
  }
  const ms = stats.models;
  const total = ms.reduce((a, m) => a + m.n, 0);
  const errors = ms.reduce((a, m) => a + (m.errors || 0), 0);
  const labelled = ms.reduce((a, m) => a + (m.labelled || 0), 0);
  const correct = ms.reduce((a, m) => a + (m.correct || 0), 0);
  const confW = ms.reduce((a, m) => a + (m.avg_confidence || 0) * (m.n - (m.errors || 0)), 0);
  const allLat = recent.filter((r) => r.latency_ms != null).map((r) => r.latency_ms).sort((a, b) => a - b);
  $("#tiles").innerHTML = [
    ["Decisions", fmt.int(total), `${ms.length} model${ms.length === 1 ? "" : "s"}`],
    ["Error rate", total ? fmt.pct(errors / total, 1) : "–", `${fmt.int(errors)} failed`],
    ["Median latency", fmt.ms(allLat[Math.floor(allLat.length / 2)]), "last 25 decisions"],
    ["Avg confidence", total - errors ? fmt.pct(confW / (total - errors)) : "–", "top option probability"],
    ["Live accuracy", labelled ? fmt.pct(correct / labelled, 1) : "–", labelled ? `${labelled} labelled` : "label decisions below"],
  ].map(([l, v, s]) => `<div class="panel tile"><div class="label">${l}</div><div class="value">${v}</div><div class="sub">${s}</div></div>`).join("");

  const common = { series: stats.series, since: stats.since, until: stats.until, bucket: stats.bucket_seconds };
  lineChart($("#chart-count"), { ...common, field: "count", format: (v) => fmt.int(v), fillZero: true });
  lineChart($("#chart-latency"), { ...common, field: "latency_ms", format: fmt.ms, fillZero: false });

  $("#model-table").innerHTML = ms.length
    ? `<thead><tr><th>Model</th><th class="num">Decisions</th><th class="num">Errors</th><th class="num">p50</th><th class="num">p95</th><th class="num">Avg confidence</th><th class="num">Coverage</th><th class="num">No-prob fallback</th><th class="num">Accuracy</th></tr></thead><tbody>` +
      ms.map((m) => `<tr><td><span class="swatch" style="background:${colorOf(m.model)}"></span>${esc(m.model)}</td>
        <td class="num">${fmt.int(m.n)}</td><td class="num">${fmt.int(m.errors)}</td><td class="num">${fmt.ms(m.p50_ms)}</td><td class="num">${fmt.ms(m.p95_ms)}</td>
        <td class="num">${fmt.pct(m.avg_confidence)}</td><td class="num">${fmt.pct(m.avg_coverage)}</td><td class="num">${fmt.int(m.constrained)}</td>
        <td class="num">${m.labelled ? fmt.pct(m.accuracy, 1) + ` <span class="muted">(${m.labelled})</span>` : "–"}</td></tr>`).join("") + "</tbody>"
    : `<tbody><tr><td class="empty">No decisions yet — try the Playground.</td></tr></tbody>`;

  $("#recent").innerHTML = recent.length
    ? `<thead><tr><th>Time</th><th>Model</th><th>Question</th><th>Answer</th><th class="num">Conf.</th><th class="num">Latency</th><th>Correct answer</th></tr></thead><tbody>` +
      recent.map((r) => {
        const opts = r.kind === "binary" ? ["Yes", "No"] : r.probs ? Object.keys(r.probs) : r.options;
        const sel = r.error ? "" : `<select data-id="${r.id}" class="fb" style="min-width:110px"><option value="">—</option>${opts.map((o) => `<option ${r.label === o ? "selected" : ""}>${esc(o)}</option>`).join("")}</select>`;
        const mark = r.label ? (r.label === r.choice ? ' <span class="status ok">right</span>' : ' <span class="status bad">wrong</span>') : "";
        return `<tr><td class="muted">${fmt.datetime(r.ts)}</td><td><span class="swatch" style="background:${colorOf(r.model)}"></span>${esc(r.model)}</td>
          <td class="q" title="${esc(r.question)}">${esc(r.question)}</td>
          <td>${r.error ? `<span class="err" title="${esc(r.error)}">error</span>` : esc(r.choice)}</td>
          <td class="num">${fmt.pct(r.confidence)}</td><td class="num">${fmt.ms(r.latency_ms)}</td><td>${sel}${mark}</td></tr>`;
      }).join("") + "</tbody>"
    : `<tbody><tr><td class="empty">Nothing logged in this range.</td></tr></tbody>`;
  $$("#recent select.fb").forEach((s) =>
    s.addEventListener("change", async () => {
      try {
        await api(`/api/decisions/${s.dataset.id}/feedback`, { method: "POST", body: { label: s.value || null } });
      } catch (e) { toast(e.message, true); }
      refreshOverview();
    })
  );
}

$("#auto").addEventListener("change", schedule);
function schedule() {
  clearInterval(state.timer);
  if ($("#auto").checked) state.timer = setInterval(() => { if (!document.hidden) refreshOverview(); }, 5000);
}

// ---------------------------------------------------------------- models
async function loadModels() {
  state.models = await api("/api/models");
  const rows = state.models.map((m) => `<tr data-name="${esc(m.name)}">
      <td><span class="swatch" style="background:${colorOf(m.name)}"></span><b>${esc(m.name)}</b></td>
      <td>${esc(m.backend)}</td><td><code>${esc(m.config.model)}</code></td>
      <td class="num"><input type="number" step="0.05" min="0.05" max="50" value="${m.temperature}" style="width:80px" class="temp"></td>
      <td class="st muted">–</td>
      <td style="white-space:nowrap"><button class="btn small test">Test</button> <button class="btn small del">Remove</button></td></tr>`);
  $("#models-table").innerHTML = state.models.length
    ? `<thead><tr><th>Name</th><th>Backend</th><th>Model</th><th class="num">Calibration temp.</th><th>Status</th><th></th></tr></thead><tbody>${rows.join("")}</tbody>`
    : `<tbody><tr><td class="empty">No models yet. Add one below.</td></tr></tbody>`;
  $$("#models-table tr[data-name]").forEach((tr) => {
    const name = tr.dataset.name;
    $(".test", tr).addEventListener("click", async (e) => {
      const st = $(".st", tr);
      e.target.disabled = true;
      st.innerHTML = `<span class="status warn">testing… (first call loads the model)</span>`;
      try {
        const r = await api(`/api/models/${encodeURIComponent(name)}/test`, { method: "POST" });
        if (!r.ok) st.innerHTML = `<span class="status bad">${esc(r.error)}</span>`;
        else if (r.probe?.error) st.innerHTML = `<span class="status bad">${esc(r.probe.error)}</span>`;
        else st.innerHTML = `<span class="status ok">ok · ${esc(r.probe.method)} · ${fmt.ms(r.probe.latency_ms)} · P(yes)=${fmt.num(r.probe.score, 2)}</span>`;
      } catch (err) {
        st.innerHTML = `<span class="status bad">${esc(err.message)}</span>`;
      }
      e.target.disabled = false;
    });
    $(".del", tr).addEventListener("click", async () => {
      if (!confirm(`Remove model "${name}"? Its logged decisions are kept.`)) return;
      try {
        await api(`/api/models/${encodeURIComponent(name)}`, { method: "DELETE" });
      } catch (err) { toast(err.message, true); }
      await loadModels();
    });
    $(".temp", tr).addEventListener("change", async (e) => {
      try {
        await api(`/api/models/${encodeURIComponent(name)}/temperature`, { method: "PUT", body: { temperature: +e.target.value } });
        toast(`${name}: temperature set to ${e.target.value}`);
      } catch (err) { toast(err.message, true); }
    });
  });
  const checks = (id) => {
    const prev = new Set($$(`#${id} input:checked`).map((i) => i.value));
    $("#" + id).innerHTML = state.models.length
      ? state.models.map((m, i) => `<label><input type="checkbox" value="${esc(m.name)}" ${prev.size ? (prev.has(m.name) ? "checked" : "") : i === 0 ? "checked" : ""}><span class="swatch" style="background:${colorOf(m.name)}"></span>${esc(m.name)}</label>`).join("")
      : `<span class="help">Register a model on the Models tab first.</span>`;
  };
  checks("pg-models");
  checks("ev-models");
}

// One-click settings for common providers. Notes say whether real probabilities come back.
const PRESETS = [
  { label: "Ollama (local)", name: "local", backend: "ollama", config: { model: "qwen3:4b", host: "http://127.0.0.1:11434" },
    note: "Real probabilities (Ollama 0.12.11+). Pull the model first: ollama pull qwen3:4b" },
  { label: "OpenAI", name: "openai", backend: "openai", config: { model: "gpt-4.1-mini", base_url: "https://api.openai.com/v1", api_key_env: "OPENAI_API_KEY" },
    note: "gpt-4.1 / gpt-4o give real probabilities; gpt-5.1+ and GPT-6 get reasoning_effort=none automatically; gpt-5 and o-series fall back to plain answers." },
  { label: "OpenRouter", name: "openrouter", backend: "openai", config: { model: "openai/gpt-4.1-mini", base_url: "https://openrouter.ai/api/v1", api_key_env: "OPENROUTER_API_KEY" },
    note: "Probabilities depend on the provider behind the model; others fall back to plain answers." },
  { label: "vLLM", name: "vllm", backend: "openai", config: { model: "Qwen/Qwen2.5-7B-Instruct", base_url: "http://localhost:8000/v1", api_key_env: "VLLM_API_KEY", prefill: true },
    note: "Real probabilities, with the 'Answer:' prefill." },
  { label: "llama.cpp server", name: "llamacpp", backend: "openai", config: { model: "local", base_url: "http://localhost:8080/v1", api_key_env: "LLAMACPP_API_KEY" },
    note: "Real probabilities from llama-server." },
  { label: "LM Studio", name: "lmstudio", backend: "openai", config: { model: "qwen2.5-7b-instruct", base_url: "http://localhost:1234/v1", api_key_env: "LMSTUDIO_API_KEY" },
    note: "LM Studio's chat endpoint returns no probabilities, so answers are plain (constrained)." },
  { label: "Groq", name: "groq", backend: "openai", config: { model: "llama-3.3-70b-versatile", base_url: "https://api.groq.com/openai/v1", api_key_env: "GROQ_API_KEY" },
    note: "Groq returns no probabilities: plain answers only." },
  { label: "Jev (TypeSafe)", name: "jev", backend: "systemone", config: { model: "jev-latest", base_url: "https://api.typesafe.ai/v1", api_key_env: "TYPESAFE_API_KEY" },
    note: "Hosted decision model; Verdict adds calibration, debiasing and evaluation on top." },
  { label: "Ollama decision models", name: "ollama-decide", backend: "systemone", config: { model: "tev1:4b", base_url: "http://127.0.0.1:11434/v1" },
    note: "Ollama 0.35+ decision models behind /v1/systemone." },
  { label: "Another Verdict server", name: "remote-verdict", backend: "systemone", config: { model: "jev-latest", base_url: "http://other-host:8420/v1", api_key_env: "VERDICT_REMOTE_API_KEY" },
    note: "Uses that server's /v1/systemone endpoint; its --token goes in the env var." },
  { label: "Demo (no model)", name: "demo", backend: "mock", config: { model: "mock" },
    note: "A fake model for trying the dashboard." },
];

function applyPreset() {
  const p = PRESETS[+$("#m-preset").value - 1];
  if (!p) { $("#m-preset-note").textContent = "Pick a provider to fill in the settings."; return; }
  $("#m-backend").value = p.backend;
  renderBackendFields();
  $$("#m-fields [data-key]").forEach((i) => {
    if (i.dataset.key in p.config) i.value = String(p.config[i.dataset.key]);
    else if (i.dataset.key === "api_key_env") i.value = "";
  });
  $("#m-name").value = p.name;
  $("#m-preset-note").textContent = p.note;
}
$("#m-preset").innerHTML += PRESETS.map((p, i) => `<option value="${i + 1}">${esc(p.label)}</option>`).join("");
$("#m-preset").addEventListener("change", applyPreset);

function renderBackendFields() {
  const b = $("#m-backend").value;
  const fields = state.backends[b] || {};
  $("#m-fields").innerHTML = Object.entries(fields).map(([k, f]) => {
    const input = f.type === "bool"
      ? `<select data-key="${k}" data-type="bool"><option value="true" ${f.default ? "selected" : ""}>true</option><option value="false" ${!f.default ? "selected" : ""}>false</option></select>`
      : `<input type="${f.type === "float" ? "number" : "text"}" data-key="${k}" data-type="${f.type}" value="${esc(f.default)}">`;
    return `<label class="field">${esc(k)} ${input}<span class="help">${esc(f.help)}</span></label>`;
  }).join("");
}
$("#m-backend").addEventListener("change", renderBackendFields);
$("#m-add").addEventListener("click", async () => {
  const config = {};
  $$("#m-fields [data-key]").forEach((i) => {
    const v = i.value.trim();
    if (v === "") return;
    config[i.dataset.key] = i.dataset.type === "bool" ? v === "true" : i.dataset.type === "float" ? +v : v;
  });
  const name = $("#m-name").value.trim() || config.model;
  try {
    await api("/api/models", { method: "POST", body: { name, backend: $("#m-backend").value, config } });
    toast(`Saved ${name}`);
    $("#m-name").value = "";
    await loadModels();
  } catch (e) { toast(e.message, true); }
});

// ---------------------------------------------------------------- playground
$("#pg-kind").addEventListener("change", () => {
  const k = $("#pg-kind").value;
  $("#pg-options-wrap").hidden = k !== "choice";
  $("#pg-scale-wrap").hidden = k !== "score";
});

$("#pg-run").addEventListener("click", async () => {
  const models = $$("#pg-models input:checked").map((i) => i.value);
  if (!models.length) return toast("Pick at least one model", true);
  const kind = $("#pg-kind").value;
  const body = {
    models, kind,
    question: $("#pg-question").value,
    context: $("#pg-context").value || null,
    debias: { true: true, false: false, rotate: "rotate" }[$("#pg-debias").value],
  };
  if (kind === "choice") body.options = $("#pg-options").value.split("\n").map((s) => s.trim()).filter(Boolean);
  if (kind === "score") body.scale = $("#pg-scale").value.split("-").map(Number);
  const btn = $("#pg-run");
  btn.disabled = true;
  $("#pg-results").classList.add("loading");
  $("#pg-status").textContent = "Running… local models can take a while on the first call.";
  try {
    const res = await api("/api/compare", { method: "POST", body });
    $("#pg-results").innerHTML = res.map(renderResult).join("");
    $("#pg-status").textContent = "";
  } catch (e) {
    toast(e.message, true);
    $("#pg-status").textContent = "";
  }
  btn.disabled = false;
  $("#pg-results").classList.remove("loading");
});

function renderResult(r) {
  if (r.error) return `<div class="panel"><h2><span class="swatch" style="background:${colorOf(r.model)}"></span>${esc(r.model)}</h2><div class="err">${esc(r.error)}</div></div>`;
  const entries = Object.entries(r.probs);
  const headline = r.kind === "binary" ? `P(yes) = ${fmt.pct(r.score, 1)}`
    : r.kind === "score" ? `Score ${fmt.num(r.value, 2)} <span class="muted" style="font-size:13px;font-weight:400">(${fmt.pct(r.score)} of scale)</span>`
    : `${esc(r.choice)} <span class="muted" style="font-size:13px;font-weight:400">${fmt.pct(r.confidence, 1)}</span>`;
  const bars = entries.map(([o, p]) => `<div class="opt ${o === r.choice ? "win" : ""}" title="${esc(o)}">${esc(o)}</div>
      <div class="track"><div class="fill" style="width:${(p * 100).toFixed(2)}%;background:${colorOf(r.model)}"></div></div>
      <div class="pct">${fmt.pct(p, 1)}</div>`).join("");
  const warn = r.coverage != null && r.coverage < 0.5 ? `<div class="help" style="margin-top:8px"><span class="status warn">Low coverage</span> — the model mostly wanted to say something other than a valid label; treat with care.</div>` : "";
  const fb = r.method === "constrained" ? `<div class="help" style="margin-top:8px"><span class="status warn">No probabilities</span> — backend did not return logprobs; used constrained output.</div>` : "";
  return `<div class="panel"><h2><span class="swatch" style="background:${colorOf(r.model)}"></span>${esc(r.model)}</h2>
    <div class="verdict-line">${headline}</div><div class="bars">${bars}</div>
    <div class="kv"><span>latency <b>${fmt.ms(r.latency_ms)}</b></span><span>method <b>${esc(r.method)}</b></span>
      <span>coverage <b>${fmt.pct(r.coverage, 1)}</b></span><span>temp <b>${r.temperature}</b></span></div>${warn}${fb}
    ${r.top_tokens?.length ? `<details style="margin-top:8px"><summary class="help" style="cursor:pointer">Raw top tokens</summary><div class="kv">${r.top_tokens.map(([t, lp]) => `<span><code>${esc(JSON.stringify(t))}</code> ${fmt.pct(Math.exp(lp), 1)}</span>`).join("")}</div></details>` : ""}
  </div>`;
}

// ---------------------------------------------------------------- evaluate
$("#ev-data").value = [
  { context: "Your invoice #4411 for March is attached. Payment due in 30 days.", question: "Which folder?", options: ["Work", "Personal", "Spam"], answer: "Work" },
  { context: "Congratulations!!! You were selected for a FREE iPhone. Reply with your bank details.", question: "Which folder?", options: ["Work", "Personal", "Spam"], answer: "Spam" },
  { context: "Hey, are we still on for dinner at mum's on Sunday?", question: "Which folder?", options: ["Work", "Personal", "Spam"], answer: "Personal" },
  { context: "The server is down and customers cannot log in.", question: "Is this urgent?", kind: "binary", answer: "Yes" },
  { context: "Weekly newsletter: 5 tips for better sleep.", question: "Is this urgent?", kind: "binary", answer: "No" },
].map((o) => JSON.stringify(o)).join("\n");

$("#ev-load").addEventListener("click", async () => {
  try {
    const r = await fetch("/static/benchmark.jsonl");
    if (!r.ok) throw new Error(r.statusText);
    $("#ev-data").value = (await r.text()).trim();
    toast("Loaded 36 labelled examples: email folders, support routing, urgency, sarcasm and review scores");
  } catch (e) { toast("Could not load the benchmark: " + e.message, true); }
});

$("#ev-run").addEventListener("click", async () => {
  const models = $$("#ev-models input:checked").map((i) => i.value);
  if (!models.length) return toast("Pick at least one model", true);
  let examples;
  try {
    examples = $("#ev-data").value.split("\n").map((l) => l.trim()).filter(Boolean).map((l) => JSON.parse(l));
  } catch (e) { return toast("Invalid JSON line: " + e.message, true); }
  const btn = $("#ev-run");
  btn.disabled = true;
  $("#ev-status").textContent = `Running ${examples.length} examples × ${models.length} model(s)…`;
  try {
    const res = await api("/api/evaluate", { method: "POST", body: { models, examples, log: $("#ev-log").checked } });
    $("#ev-results-panel").hidden = false;
    $("#ev-results").innerHTML = `<thead><tr><th>Model</th><th class="num">Examples</th><th class="num">Errors</th><th class="num">Accuracy</th>
      <th class="num">Log loss raw → calibrated</th><th class="num">ECE raw → calibrated</th><th class="num">Fitted temp.</th><th class="num">Avg latency</th><th></th></tr></thead><tbody>` +
      res.map((r) => `<tr><td><span class="swatch" style="background:${colorOf(r.model)}"></span>${esc(r.model)}</td>
        <td class="num">${r.n}</td><td class="num">${r.error_count ? `<span class="err" title="${esc(r.errors.join("\n"))}">${r.error_count}</span>` : 0}</td>
        <td class="num">${fmt.pct(r.raw.accuracy, 1)}</td>
        <td class="num">${fmt.num(r.raw.log_loss)} → ${fmt.num(r.calibrated.log_loss)}</td>
        <td class="num">${fmt.num(r.raw.ece)} → ${fmt.num(r.calibrated.ece)}</td>
        <td class="num">${r.fitted_temperature}</td><td class="num">${fmt.ms(r.avg_latency_ms)}</td>
        <td>${r.reliable
          ? `<button class="btn small apply" data-model="${esc(r.model)}" data-t="${r.fitted_temperature}">Apply temp.</button>`
          : `<button class="btn small" disabled title="${esc(r.warning || "")}">Apply temp.</button>`}</td></tr>
        ${r.reliable ? "" : `<tr><td colspan="9" class="help"><span class="status warn">Not enough evidence</span> ${esc(r.model)}: ${esc(r.warning || "")}</td></tr>`}`).join("") + "</tbody>";
    $$("#ev-results .apply").forEach((b) => b.addEventListener("click", async () => {
      try {
        await api(`/api/models/${encodeURIComponent(b.dataset.model)}/temperature`, { method: "PUT", body: { temperature: +b.dataset.t } });
        toast(`${b.dataset.model}: calibration temperature set to ${b.dataset.t}`);
        loadModels();
      } catch (e) { toast(e.message, true); }
    }));
  } catch (e) { toast(e.message, true); }
  $("#ev-status").textContent = "";
  btn.disabled = false;
});

// ---------------------------------------------------------------- boot
(async function boot() {
  state.backends = await api("/api/backends");
  $("#m-backend").innerHTML = Object.keys(state.backends).map((k) => `<option>${k}</option>`).join("");
  renderBackendFields();
  await loadModels();
  let tab = "overview";
  try { tab = localStorage.getItem("verdict.tab") || tab; } catch {}
  $(`nav button[data-tab="${tab}"]`)?.click();
  schedule();
  window.addEventListener("resize", () => refreshOverview());
})();
