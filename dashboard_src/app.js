'use strict';
/* nanoGPT Lab Bench. Every number comes from DATA (built from saved run files by
   build_dashboard.py); every live prediction comes from the saved weights run
   through a JavaScript copy of the nanoGPT forward pass below. */

const DATA = JSON.parse(document.getElementById('data').textContent);
const RUNS = DATA.runs;
const CASES = DATA.suite.cases;
const CASE_BY_ID = Object.fromEntries(CASES.map(c => [c.id, c]));
const CHECKS = DATA.checks;
const LOCAL = location.protocol === 'file:' || /^(localhost|127\.|0\.0\.0\.0|\[::1\])/.test(location.hostname);
const REPO_URL = DATA.repo_url || null;

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const pct = (x, d = 1) => x == null || isNaN(x) ? '—' : (x * 100).toFixed(d) + '%';
const num = x => x == null ? '—' : Number(x).toLocaleString('en-US');
const fx = (x, d = 3) => x == null ? '—' : Number(x).toFixed(d);
const prob = x => x == null ? '—' : x >= 0.1 ? (x * 100).toFixed(1) + '%' : x >= 0.001 ? (x * 100).toFixed(2) + '%' : x.toExponential(1);
const sci = x => x == null ? '—' : Math.abs(x) < 1e-3 && x !== 0 ? x.toExponential(4) : Number(x).toPrecision(6);
const cssVar = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const runColor = i => `var(--s${(i % 8) + 1})`;
const STAGES = { untrained: 'Untrained', trained: 'Trained' };
const evalStageKey = st => st === 'trained' ? 'final' : 'untrained';
const store = {
  get(k, d) { try { const v = localStorage.getItem('bench:' + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem('bench:' + k, JSON.stringify(v)); } catch (e) { /* storage unavailable */ } },
};

function fileHref(path) {
  if (REPO_URL) return `${REPO_URL.replace(/\/$/, '')}/${path}`;
  return LOCAL ? path : null;
}
function fileLink(path, label) {
  const href = fileHref(path);
  return href ? `<a class="path" href="${esc(href)}" target="_blank" rel="noopener">${esc(label || path)}</a>`
    : `<code class="path" title="${esc(path)}">${esc(label || path)}</code>`;
}

/* ------------------------------------------------------------------ tokenizer */
const TOKEN_RE = /[\p{L}\p{N}\p{M}_]+(?:['’][\p{L}\p{N}\p{M}_]+)*|[^\p{L}\p{N}\p{M}_\s]/gu;
const tokenize = text => (text || '').toLowerCase().match(TOKEN_RE) || [];

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ------------------------------------------------------------------ nanoGPT forward pass
   Mirrors nanogpt_model.py: token + position embeddings, N blocks of
   LayerNorm → causal multi-head attention → residual, LayerNorm → MLP(4×, exact GELU) → residual,
   final LayerNorm, output head tied to the token embedding table. */
function buildModel(run, stage) {
  const pack = run.weights[stage];
  const bin = atob(pack.b64);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  const all = new Float32Array(u8.buffer);
  const W = {};
  for (const e of pack.index) W[e.name] = all.subarray(e.offset, e.offset + e.shape.reduce((a, b) => a * b, 1));
  const A = pack.args;
  const NE = A.n_embd, NH = A.n_head, NL = A.n_layer, BS = A.block_size, HS = NE / NH, NF = 4 * NE;
  const vocab = run.vocab, V = vocab.length;
  const stoi = new Map(vocab.map((t, i) => [t, i]));
  const UNK = stoi.get('<UNK>'), BOS = stoi.get('<BOS>'), EOS = stoi.get('<EOS>');
  const wte = W['transformer.wte.weight'], wpe = W['transformer.wpe.weight'];

  const erf = x => { const s = x < 0 ? -1 : 1; x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x);
    return s * (1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x)); };
  const gelu = x => 0.5 * x * (1 + erf(x / Math.SQRT2));
  function layerNorm(v, w, b) {
    let m = 0; for (let i = 0; i < NE; i++) m += v[i]; m /= NE;
    let s = 0; for (let i = 0; i < NE; i++) { const d = v[i] - m; s += d * d; } s /= NE;
    const inv = 1 / Math.sqrt(s + 1e-5), out = new Float64Array(NE);
    for (let i = 0; i < NE; i++) out[i] = (v[i] - m) * inv * w[i] + b[i];
    return out;
  }
  function linear(x, w, b, nIn, nOut) {
    const out = new Float64Array(nOut);
    for (let o = 0; o < nOut; o++) { let acc = b ? b[o] : 0; const base = o * nIn; for (let i = 0; i < nIn; i++) acc += w[base + i] * x[i]; out[o] = acc; }
    return out;
  }
  function forward(allIds) {
    const T = Math.min(allIds.length, BS), ids = allIds.slice(allIds.length - T);
    let x = ids.map((id, t) => { const v = new Float64Array(NE); for (let i = 0; i < NE; i++) v[i] = wte[id * NE + i] + wpe[t * NE + i]; return v; });
    const attn = [], mlpLast = [];
    for (let l = 0; l < NL; l++) {
      const P = `transformer.h.${l}.`;
      const q = [], k = [], v = [];
      for (let t = 0; t < T; t++) {
        const qkv = linear(layerNorm(x[t], W[P + 'ln_1.weight'], W[P + 'ln_1.bias']), W[P + 'attn.c_attn.weight'], W[P + 'attn.c_attn.bias'], NE, 3 * NE);
        q.push(qkv.subarray(0, NE)); k.push(qkv.subarray(NE, 2 * NE)); v.push(qkv.subarray(2 * NE));
      }
      const layerAttn = Array.from({ length: NH }, () => []);
      const next = [];
      for (let t = 0; t < T; t++) {
        const y = new Float64Array(NE);
        for (let h = 0; h < NH; h++) {
          const off = h * HS, sc = new Float64Array(t + 1); let mx = -Infinity;
          for (let s = 0; s <= t; s++) { let d = 0; for (let i = 0; i < HS; i++) d += q[t][off + i] * k[s][off + i]; d /= Math.sqrt(HS); sc[s] = d; if (d > mx) mx = d; }
          let sum = 0; for (let s = 0; s <= t; s++) { sc[s] = Math.exp(sc[s] - mx); sum += sc[s]; }
          const row = new Array(t + 1);
          for (let s = 0; s <= t; s++) { const a = sc[s] / sum; row[s] = a; for (let i = 0; i < HS; i++) y[off + i] += a * v[s][off + i]; }
          layerAttn[h].push(row);
        }
        const proj = linear(y, W[P + 'attn.c_proj.weight'], W[P + 'attn.c_proj.bias'], NE, NE);
        const r = new Float64Array(NE); for (let i = 0; i < NE; i++) r[i] = x[t][i] + proj[i];
        const hdn = linear(layerNorm(r, W[P + 'ln_2.weight'], W[P + 'ln_2.bias']), W[P + 'mlp.c_fc.weight'], W[P + 'mlp.c_fc.bias'], NE, NF);
        for (let i = 0; i < NF; i++) hdn[i] = gelu(hdn[i]);
        if (t === T - 1) mlpLast.push(hdn);
        const mo = linear(hdn, W[P + 'mlp.c_proj.weight'], W[P + 'mlp.c_proj.bias'], NF, NE);
        for (let i = 0; i < NE; i++) r[i] += mo[i];
        next.push(r);
      }
      x = next; attn.push(layerAttn);
    }
    const hf = layerNorm(x[T - 1], W['transformer.ln_f.weight'], W['transformer.ln_f.bias']);
    const logits = new Float64Array(V);
    for (let t = 0; t < V; t++) { let acc = 0; const b = t * NE; for (let i = 0; i < NE; i++) acc += hf[i] * wte[b + i]; logits[t] = acc; }
    return { ids, T, logits, attn, mlpLast, truncated: allIds.length > BS };
  }
  function softmax(logits, temperature = 1, banBOS = false) {
    const out = new Float64Array(logits.length); let mx = -Infinity;
    for (let i = 0; i < logits.length; i++) if (!(banBOS && i === BOS) && logits[i] > mx) mx = logits[i];
    let sum = 0;
    for (let i = 0; i < logits.length; i++) { out[i] = banBOS && i === BOS ? 0 : Math.exp((logits[i] - mx) / temperature); sum += out[i]; }
    for (let i = 0; i < logits.length; i++) out[i] /= sum;
    return out;
  }
  function encode(text) {
    const toks = tokenize(text);
    return { toks, ids: toks.map(t => stoi.has(t) ? stoi.get(t) : UNK), unknown: [...new Set(toks.filter(t => !stoi.has(t)))] };
  }
  /* BOS is structural and banned from output; EOS ends the reply (as run_evals.generate_reply). */
  function generate(ids, { temperature = 0.8, maxTokens = 24, seed = 2026, greedy = false } = {}) {
    const rand = mulberry32(seed), cur = ids.slice(), out = [];
    let ended = false;
    for (let n = 0; n < maxTokens; n++) {
      const { logits } = forward(cur);
      let nextId;
      if (greedy) { let best = -Infinity; for (let i = 0; i < logits.length; i++) if (i !== BOS && logits[i] > best) { best = logits[i]; nextId = i; } }
      else { const p = softmax(logits, temperature, true); let r = rand(), acc = 0; nextId = p.length - 1; for (let i = 0; i < p.length; i++) { acc += p[i]; if (r < acc) { nextId = i; break; } } }
      if (nextId === EOS) { ended = true; break; }
      out.push(nextId); cur.push(nextId);
    }
    return { ids: out, text: out.map(i => vocab[i]).join(' '), ended };
  }
  return { W, NE, NH, NL, BS, V, NF, vocab, stoi, UNK, BOS, EOS, wte, forward, softmax, encode, generate };
}
const MODELS = new Map();
function model(ri = S.run, stage = S.stage) {
  const key = ri + ':' + stage;
  if (!MODELS.has(key)) MODELS.set(key, buildModel(RUNS[ri], stage));
  return MODELS.get(key);
}

/* ------------------------------------------------------------------ embedding geometry */
const GEOM = new Map();
function geometry(ri, stage) {
  const key = ri + ':' + stage;
  if (GEOM.has(key)) return GEOM.get(key);
  const m = model(ri, stage), { V, NE, wte } = m;
  const unit = new Float64Array(V * NE), norm = new Float64Array(V);
  for (let t = 0; t < V; t++) { let s = 0; for (let i = 0; i < NE; i++) s += wte[t * NE + i] ** 2; norm[t] = Math.sqrt(s); for (let i = 0; i < NE; i++) unit[t * NE + i] = wte[t * NE + i] / (norm[t] || 1); }
  const sim = new Float32Array(V * V);
  for (let a = 0; a < V; a++) for (let b = a; b < V; b++) { let d = 0; for (let i = 0; i < NE; i++) d += unit[a * NE + i] * unit[b * NE + i]; sim[a * V + b] = d; sim[b * V + a] = d; }
  const g = { V, NE, unit, norm, sim, row: t => wte.subarray(t * NE, (t + 1) * NE) };
  GEOM.set(key, g);
  return g;
}
function neighbors(g, t, k = 10, allowed = null) {
  const out = [];
  for (let u = 0; u < g.V; u++) if (u !== t && (!allowed || allowed.has(u))) out.push([u, g.sim[t * g.V + u]]);
  out.sort((a, b) => b[1] - a[1]);
  return out.slice(0, k);
}
function pca2(g, ids) {
  const { NE, unit } = g, n = ids.length, mean = new Float64Array(NE);
  ids.forEach(t => { for (let i = 0; i < NE; i++) mean[i] += unit[t * NE + i] / n; });
  const C = new Float64Array(NE * NE);
  ids.forEach(t => { for (let i = 0; i < NE; i++) { const di = unit[t * NE + i] - mean[i]; for (let j = 0; j < NE; j++) C[i * NE + j] += di * (unit[t * NE + j] - mean[j]); } });
  const comps = [];
  for (let c = 0; c < 2; c++) {
    let v = new Float64Array(NE).map((_, i) => Math.sin(i * 1.7 + c * 3.1) + 0.5);
    for (let it = 0; it < 120; it++) {
      const w = new Float64Array(NE);
      for (let i = 0; i < NE; i++) for (let j = 0; j < NE; j++) w[i] += C[i * NE + j] * v[j];
      comps.forEach(p => { let d = 0; for (let i = 0; i < NE; i++) d += w[i] * p[i]; for (let i = 0; i < NE; i++) w[i] -= d * p[i]; });
      const nn = Math.hypot(...w) || 1; v = w.map(x => x / nn);
    }
    comps.push(v);
  }
  const pos = new Map();
  ids.forEach(t => pos.set(t, comps.map(p => { let d = 0; for (let i = 0; i < NE; i++) d += (unit[t * NE + i] - mean[i]) * p[i]; return d; })));
  return pos;
}
function kmeans(g, ids, k = 8, seed = 7) {
  const { NE, unit } = g, rand = mulberry32(seed);
  const vec = t => unit.subarray(t * NE, (t + 1) * NE);
  const d2 = (a, b) => { let s = 0; for (let i = 0; i < NE; i++) s += (a[i] - b[i]) ** 2; return s; };
  const cents = [Float64Array.from(vec(ids[Math.floor(rand() * ids.length)]))];
  while (cents.length < k) {
    const ds = ids.map(t => Math.min(...cents.map(c => d2(vec(t), c)))), tot = ds.reduce((a, b) => a + b, 0);
    let r = rand() * tot, pick = ids[ids.length - 1];
    for (let i = 0; i < ids.length; i++) { r -= ds[i]; if (r <= 0) { pick = ids[i]; break; } }
    cents.push(Float64Array.from(vec(pick)));
  }
  const assign = new Map();
  for (let it = 0; it < 40; it++) {
    ids.forEach(t => { let best = 0, bd = Infinity; cents.forEach((c, j) => { const d = d2(vec(t), c); if (d < bd) { bd = d; best = j; } }); assign.set(t, best); });
    cents.forEach((c, j) => { const mem = ids.filter(t => assign.get(t) === j); if (!mem.length) return; c.fill(0); mem.forEach(t => { const v = vec(t); for (let i = 0; i < NE; i++) c[i] += v[i] / mem.length; }); });
  }
  return assign;
}

/* ------------------------------------------------------------------ state + routing */
const defaultRun = (() => { const idx = RUNS.map((r, i) => [r, i]).filter(([r]) => r.kind !== 'starter'); return idx.length ? idx[idx.length - 1][1] : RUNS.length - 1; })();
const S = {
  run: Math.min(store.get('run', defaultRun), RUNS.length - 1),
  stage: store.get('stage', 'trained'),
  section: 'overview',
  rendered: {},
};
if (RUNS[S.run] == null) S.run = defaultRun;
const SECTIONS = ['overview', 'network', 'lab', 'chat', 'training', 'evals', 'corpus', 'checklist', 'glossary'];
const RENDER = {};
const run = () => RUNS[S.run];

function go(section, focusId, after) {
  if (!SECTIONS.includes(section)) section = 'overview';
  S.section = section;
  $$('.section').forEach(el => { el.hidden = el.dataset.section !== section; });
  $$('.rail a').forEach(a => a.setAttribute('aria-current', a.dataset.go === section ? 'page' : 'false'));
  const key = section + ':' + S.run + ':' + S.stage;
  if (S.rendered[section] !== key) { RENDER[section](); S.rendered[section] = key; }
  if (location.hash.slice(1) !== section) history.replaceState(null, '', '#' + section);
  requestAnimationFrame(() => {
    if (after) after();
    const el = focusId && document.getElementById(focusId);
    if (el) { el.scrollIntoView({ block: 'start', behavior: 'smooth' }); el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash'); }
    else if (!after) window.scrollTo({ top: 0 });
  });
}
function rerenderCurrent() { S.rendered = {}; go(S.section); }

function setupShell() {
  const sel = $('#run-select');
  sel.innerHTML = RUNS.map((r, i) => `<option value="${i}">${esc(r.label)} · ${esc(r.id.slice(0, 15))}</option>`).join('');
  sel.value = S.run;
  sel.addEventListener('change', () => { S.run = +sel.value; store.set('run', S.run); rerenderCurrent(); });
  $$('.seg [data-stage]').forEach(b => b.addEventListener('click', () => setStage(b.dataset.stage)));
  syncStageButtons();
  $$('[data-go]').forEach(a => a.addEventListener('click', e => { e.preventDefault(); go(a.dataset.go); }));
  $('#rail-foot').innerHTML = `${RUNS.length} runs loaded<br>Built ${esc(DATA.generated.replace('T', ' ').slice(0, 16))}<br>Model runs live in this page`;
  window.addEventListener('hashchange', () => { const s = location.hash.slice(1); if (SECTIONS.includes(s) && s !== S.section) go(s); });
  const mq = window.matchMedia('(prefers-color-scheme: dark)');
  const onTheme = () => { S.rendered = {}; if (['network', 'lab', 'training', 'evals'].includes(S.section)) go(S.section); };
  mq.addEventListener?.('change', onTheme);
  new MutationObserver(onTheme).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
}
function setStage(stage) {
  if (stage === S.stage) return;
  S.stage = stage; store.set('stage', stage); syncStageButtons();
  if (S.section === 'network' && NET.graph && NET.graph.ri === S.run) { NET.graph.setStage(stage); S.rendered.network = 'network:' + S.run + ':' + S.stage; }
  else rerenderCurrent();
}
function syncStageButtons() { $$('.seg [data-stage]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.stage === S.stage))); }

/* ------------------------------------------------------------------ tooltip */
const TIP = $('#tooltip');
function tip(e, html) { TIP.innerHTML = html; TIP.hidden = false; const pad = 14, w = TIP.offsetWidth, h = TIP.offsetHeight;
  let x = e.clientX + pad, y = e.clientY + pad; if (x + w > innerWidth - 8) x = e.clientX - w - pad; if (y + h > innerHeight - 8) y = e.clientY - h - pad;
  TIP.style.left = x + 'px'; TIP.style.top = y + 'px'; }
function untip() { TIP.hidden = true; }

/* ------------------------------------------------------------------ shared bits */
function tokChips(m, ids, { showIds = true, unknownWords = [] } = {}) {
  return `<div class="toks">${ids.map(id => {
    const w = m.vocab[id], special = w.startsWith('<') && w.endsWith('>');
    return `<span class="tok${id === m.UNK ? ' unk' : ''}${special && id !== m.UNK ? ' special' : ''}">${esc(w)}${showIds ? `<sub>${id}</sub>` : ''}</span>`;
  }).join('')}</div>${unknownWords.length ? `<p class="note" style="margin-top:6px">Unknown to this model's vocabulary, fed in as <code>&lt;UNK&gt;</code>: ${unknownWords.map(w => `<b class="mono">${esc(w)}</b>`).join(', ')}</p>` : ''}`;
}
function evalStats(r, stage) { return r.evals[stage]?.summary?.overall; }
function stripCanvas(values, lo, hi) {
  const c = document.createElement('canvas'); c.width = values.length; c.height = 1; c.className = 'vec-strip';
  const ctx = c.getContext('2d'), neg = d3.color(cssVar('--s1')), pos = d3.color(cssVar('--s8')), mid = d3.color(cssVar('--panel-2'));
  const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
  values.forEach((v, i) => { const t = Math.max(-1, Math.min(1, v / m)); ctx.fillStyle = t < 0 ? d3.interpolateRgb(mid, neg)(-t) : d3.interpolateRgb(mid, pos)(t); ctx.fillRect(i, 0, 1, 1); });
  c.addEventListener('mousemove', e => { const rect = c.getBoundingClientRect(); const i = Math.min(values.length - 1, Math.floor((e.clientX - rect.left) / rect.width * values.length)); tip(e, `dimension <b>${i}</b> = <b>${values[i].toFixed(5)}</b>`); });
  c.addEventListener('mouseleave', untip);
  return c;
}
function seqColor(t) { return d3.interpolateRgb(cssVar('--seq-0'), cssVar('--seq-3'))(Math.max(0, Math.min(1, t))); }
function inkOn(bg) { const c = d3.rgb(bg); const L = (0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b) / 255; return L > 0.55 ? '#15212b' : '#ffffff'; }
function probBars(rows, { color = 'var(--accent)', ghostLabel = '' } = {}) {
  const max = Math.max(...rows.map(r => Math.max(r.p, r.ghost ?? 0)), 1e-9);
  return `<div class="probbars">${rows.map(r => `
    <span class="w">${esc(r.w)}</span>
    <span class="track" data-tip="${esc(`<b>${esc(r.w)}</b>: ${prob(r.p)}${r.ghost != null ? `<br>${esc(ghostLabel)}: ${prob(r.ghost)}` : ''}`)}">
      <span class="fill" style="width:${(r.p / max * 100).toFixed(2)}%;background:${r.color || color}"></span>
      ${r.ghost != null ? `<span class="ghost" style="left:calc(${(r.ghost / max * 100).toFixed(2)}% - 1px)"></span>` : ''}
    </span>
    <span class="num mono" style="font-size:12px">${prob(r.p)}</span>`).join('')}</div>`;
}
function bindTips(root) {
  $$('[data-tip]', root).forEach(el => { el.addEventListener('mousemove', e => tip(e, el.dataset.tip)); el.addEventListener('mouseleave', untip); });
}
function topK(p, k, exclude = new Set()) {
  const idx = []; for (let i = 0; i < p.length; i++) if (!exclude.has(i)) idx.push(i);
  idx.sort((a, b) => p[b] - p[a]); return idx.slice(0, k);
}

/* ================================================================== OVERVIEW */
RENDER.overview = function () {
  const r = run(), c = r.config, sec = $('#sec-overview');
  const req = requirementStatus();
  const counts = { done: 0, you: 0, todo: 0 }; req.flatMap(g => g.items).forEach(i => counts[i.status]++);
  const total = counts.done + counts.you + counts.todo;
  sec.innerHTML = `
  <div class="hero">
    <div class="hero-copy">
      <span class="eyebrow">Class 4 · Building a custom LLM</span>
      <h1>A ${num(c.parameters)}-parameter language model, opened up for inspection</h1>
      <p>Every run in <code>llm_runs/</code> is loaded here: losses, samples, all 48 eval cases, the corpus and the saved weights. The nanoGPT forward pass runs live in this page, so the word network, attention maps and chat come from the real trained model, not from canned text.</p>
      <div class="spec">
        <span>${c.n_layer} blocks</span><span>${c.n_head} heads</span><span>${c.n_embd}-number embeddings</span><span>${c.block_size}-token context</span>
        <span>${c.vocabulary_size}-word vocabulary</span><span>${num(c.training_steps)} steps</span><span>lr ${c.learning_rate}</span><span>seed ${c.seed}</span>
      </div>
      <div class="row" style="margin-top:6px">
        <button class="btn primary" data-go2="network">Explore the word network</button>
        <button class="btn" data-go2="lab">Watch attention on a prompt</button>
        <button class="btn" data-go2="checklist">Assignment checklist · ${counts.done}/${total} done</button>
      </div>
    </div>
    <div class="panel pipeline" aria-label="Data flow">${pipelineSVG()}</div>
  </div>

  <div class="panel" id="p-overview-runs">
    <div class="panel-head"><h2>Experiments</h2><p>All-case eval success out of 48, before and after training. Losses are not comparable across corpora.</p></div>
    <div class="scroll-x"><table>
      <thead><tr><th>Run</th><th>Corpus</th><th class="n">Passages</th><th class="n">Vocab</th><th class="n">Val loss (final)</th><th>Eval: untrained → trained</th><th class="n">Scorable</th><th class="n">Steps · time</th></tr></thead>
      <tbody>${RUNS.map((rr, i) => {
        const u = evalStats(rr, 'untrained'), f = evalStats(rr, 'final'), last = rr.history[rr.history.length - 1];
        return `<tr class="clickable" data-run="${i}">
          <td><span class="swatch" style="background:${runColor(i)}"></span> <b>${esc(rr.label)}</b><br><span class="path">${esc(rr.id)}</span></td>
          <td>${rr.manifest.files.length ? rr.manifest.files.map(f => `<code>${esc(f.file)}</code>`).join(', ') : 'classroom sentences only'}</td>
          <td class="n">${num(rr.manifest.unique_passages)}</td><td class="n">${rr.config.vocabulary_size}</td>
          <td class="n">${fx(last?.validation_loss)}</td>
          <td>${u && f ? `<div class="row" style="gap:8px;flex-wrap:nowrap"><span class="mono num">${u.correct}</span><div class="bar-track" style="width:120px;position:relative"><div class="bar-fill" style="width:${f.correct / f.total * 100}%;background:${runColor(i)}"></div><div style="position:absolute;top:-2px;bottom:-2px;width:2px;background:var(--ink);left:${u.correct / u.total * 100}%"></div></div><b class="mono num">${f.correct}</b><span class="faint">/ ${f.total}</span></div>` : '<span class="faint">no evals</span>'}</td>
          <td class="n">${f ? f.scorable : '—'}</td>
          <td class="n">${num(rr.summary?.completed_steps)} · ${rr.summary ? fx(rr.summary.elapsed_seconds, 0) + ' s' : '—'}</td></tr>`;
      }).join('')}</tbody></table></div>
    <p class="note" style="margin-top:10px">Bar = trained score; black tick = untrained score. Click a row to make it the active run for every view.</p>
  </div>

  <div class="grid g2">
    <div class="panel">
      <div class="panel-head"><h2>Where the assignment stands</h2></div>
      <div class="progress" aria-hidden="true"><i style="width:${counts.done / total * 100}%;background:var(--good)"></i><i style="width:${counts.you / total * 100}%;background:var(--warn)"></i><i style="width:${counts.todo / total * 100}%;background:var(--bad)"></i></div>
      <div class="row" style="margin-top:10px"><span class="pill done">${counts.done} done</span><span class="pill you">${counts.you} need your writing</span><span class="pill todo">${counts.todo} not yet</span></div>
      <ul class="clean" style="margin-top:12px">${req.flatMap(g => g.items).filter(i => i.status !== 'done').map(i => `<li><a href="#checklist" data-req="${esc(i.key)}">${esc(i.t)}</a> <span class="pill ${i.status}">${i.status === 'you' ? 'your words' : 'not yet'}</span></li>`).join('') || '<li>Everything the dashboard can check is done.</li>'}</ul>
    </div>
    <div class="panel">
      <div class="panel-head"><h2>Adding your own data</h2></div>
      <ol class="clean next-steps">
        <li>Your ZIP files are unpacked into <code>corpus/&lt;name&gt;/</code>. PDF, TXT and Markdown are read. Nothing goes near <code>evals/</code>.</li>
        <li>Before training, every one of the 48 eval prompts is searched for in the new text. Any match stops the run.</li>
        <li>A fresh model is trained on the classroom sentences plus your files, then all 48 cases are rerun untrained and trained.</li>
        <li>The dashboard is rebuilt with <code>python build_dashboard.py</code>. The new run appears in the Run menu above.</li>
      </ol>
      <p class="note" style="margin-top:10px">The vocabulary is capped at 509 word types. Large documents push rarer words to <code>&lt;UNK&gt;</code>. Check the unknown-token rates on the Corpus page after each run.</p>
    </div>
  </div>`;
  $$('[data-go2]', sec).forEach(b => b.addEventListener('click', () => go(b.dataset.go2)));
  $$('tr[data-run]', sec).forEach(tr => tr.addEventListener('click', () => { S.run = +tr.dataset.run; $('#run-select').value = S.run; store.set('run', S.run); rerenderCurrent(); }));
  $$('[data-req]', sec).forEach(a => a.addEventListener('click', e => { e.preventDefault(); go('checklist', 'req-' + a.dataset.req); }));
};

function pipelineSVG() {
  const box = (x, y, w, t, s, strong) => `<g><rect x="${x}" y="${y}" width="${w}" height="46" rx="6" style="fill:${strong ? 'var(--accent-weak)' : 'var(--panel)'};stroke:${strong ? 'var(--accent)' : 'var(--line-2)'}"/>
    <text x="${x + 10}" y="${y + 19}" style="font:600 12px var(--f-display);fill:var(--ink)">${t}</text><text x="${x + 10}" y="${y + 35}" style="font:400 10.5px var(--f-mono);fill:var(--ink-2)">${s}</text></g>`;
  const arrow = (x1, y1, x2, y2) => `<path d="M${x1} ${y1} L${x2} ${y2}" style="stroke:var(--ink-3);stroke-width:1.4;fill:none" marker-end="url(#ah)"/>`;
  const r = run(), c = r.config;
  return `<svg viewBox="0 0 440 250" role="img" aria-label="Pipeline from corpus to model to evals">
    <defs><marker id="ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L8 4 L0 8 z" style="fill:var(--ink-3)"/></marker></defs>
    <text x="4" y="14" style="font:600 10.5px var(--f-display);letter-spacing:.08em;fill:var(--ink-3)">TRAINING PATH</text>
    ${box(4, 24, 128, 'corpus/ + classroom', `${num(r.manifest.unique_passages)} passages`)}
    ${arrow(132, 47, 152, 47)}
    ${box(154, 24, 128, 'vocabulary', `${c.vocabulary_size} word types`)}
    ${arrow(282, 47, 302, 47)}
    ${box(304, 24, 132, 'AdamW training', `${num(c.training_steps)} steps`)}
    ${arrow(370, 70, 370, 96)}
    ${box(304, 98, 132, 'model.pt', `${num(c.parameters)} params`, true)}
    <line x1="4" y1="166" x2="436" y2="166" style="stroke:var(--bad);stroke-width:1.2;stroke-dasharray:5 4"/>
    <text x="4" y="160" style="font:600 10.5px var(--f-display);letter-spacing:.08em;fill:var(--bad)">NEVER CROSSED: EVAL TEXT STAYS OUT OF TRAINING</text>
    <text x="4" y="186" style="font:600 10.5px var(--f-display);letter-spacing:.08em;fill:var(--ink-3)">EVALUATION PATH</text>
    ${box(4, 196, 150, 'evals/ · 48 cases', 'prompt prefix only')}
    ${arrow(154, 219, 300, 219)}
    <path d="M370 144 L370 194" style="stroke:var(--accent);stroke-width:1.4;fill:none" marker-end="url(#ah)"/>
    ${box(304, 196, 132, 'score + continuation', 'answer key used after')}
  </svg>`;
}

/* ================================================================== WORD NETWORK */
const NET = { graph: null, k: store.get('net.k', 3), min: store.get('net.min', 0.25), color: store.get('net.color', 'cluster'), special: false, layout: store.get('net.layout', 'force'), pinDrag: true, selected: null, filter: '' };
RENDER.network = function () {
  const sec = $('#sec-network'), r = run();
  sec.innerHTML = `
  <div class="sec-head"><div class="intro"><span class="eyebrow">Embeddings · ${esc(r.label)}</span><h1>Word network</h1>
    <p>Every word is a point. Lines join each word to its closest words by cosine similarity of their 64-number embedding vectors. Drag words, scroll to zoom, click one to inspect it. Switch between <b>Untrained</b> and <b>Trained</b> at the top and watch random vectors organise into related groups.</p></div></div>
  <div class="panel" style="padding:12px 14px">
    <div class="controls">
      <label class="ctl" for="net-find">Find <input id="net-find" type="search" placeholder="word…" list="net-words" style="width:130px"></label>
      <datalist id="net-words">${r.vocab.map(w => `<option value="${esc(w)}">`).join('')}</datalist>
      <label class="ctl" for="net-k">Neighbours per word <input id="net-k" type="range" min="1" max="8" step="1" value="${NET.k}"><output id="net-k-o">${NET.k}</output></label>
      <label class="ctl" for="net-min">Min similarity <input id="net-min" type="range" min="0" max="0.9" step="0.05" value="${NET.min}"><output id="net-min-o">${NET.min.toFixed(2)}</output></label>
      <label class="ctl" for="net-color">Colour by <select id="net-color">
        <option value="cluster">Clusters in trained vectors</option><option value="freq">Training frequency</option>
        <option value="moved">Distance moved in training</option><option value="eval">Words the 48 evals use</option></select></label>
      <label class="ctl" for="net-layout">Layout <select id="net-layout"><option value="force">Force (neighbours pull)</option><option value="pca">PCA map (2D projection)</option></select></label>
      <label class="ctl"><input type="checkbox" id="net-special"> Show &lt;UNK&gt; &lt;BOS&gt; &lt;EOS&gt;</label>
      <button class="btn small" id="net-reset">Reset view</button>
    </div>
  </div>
  <div class="grid g-side">
    <div class="net-wrap" id="net-wrap"><svg id="net-svg" aria-label="Word similarity network"></svg>
      <div class="net-hud"><div id="net-legend" class="legend"></div><div id="net-stat" class="mono"></div></div></div>
    <div class="panel inspector" id="net-inspector"></div>
  </div>
  <p class="note">Cosine similarity uses all 64 dimensions. The PCA layout squeezes those 64 numbers onto a flat page, so distances there are approximate. The output layer reuses these same vectors (weight tying), so words the model predicts in similar contexts end up with similar vectors.</p>`;
  $('#net-color').value = NET.color; $('#net-layout').value = NET.layout; $('#net-special').checked = NET.special;
  const upd = () => NET.graph && NET.graph.update();
  $('#net-k').addEventListener('input', e => { NET.k = +e.target.value; $('#net-k-o').textContent = NET.k; store.set('net.k', NET.k); upd(); });
  $('#net-min').addEventListener('input', e => { NET.min = +e.target.value; $('#net-min-o').textContent = NET.min.toFixed(2); store.set('net.min', NET.min); upd(); });
  $('#net-color').addEventListener('change', e => { NET.color = e.target.value; store.set('net.color', NET.color); NET.graph.recolor(); });
  $('#net-layout').addEventListener('change', e => { NET.layout = e.target.value; store.set('net.layout', NET.layout); NET.graph.relayout(); });
  $('#net-special').addEventListener('change', e => { NET.special = e.target.checked; buildGraph(); });
  $('#net-reset').addEventListener('click', () => NET.graph.resetView());
  $('#net-find').addEventListener('change', e => { const w = e.target.value.trim().toLowerCase(); if (r.vocab.includes(w)) selectWord(w, true); });
  buildGraph();
};

function buildGraph() {
  const ri = S.run, r = RUNS[ri], mT = model(ri, 'trained');
  const gT = geometry(ri, 'trained'), gU = geometry(ri, 'untrained');
  const ids = r.vocab.map((w, i) => i).filter(i => NET.special || !/^<.+>$/.test(r.vocab[i]));
  const allowed = new Set(ids);
  const clusters = kmeans(gT, ids, 8);
  const pcaT = pca2(gT, ids), pcaU = pca2(gU, ids);
  const counts = r.token_counts || r.vocab.map(() => 1);
  const maxLog = Math.log1p(Math.max(...counts));
  const moved = new Map(ids.map(t => { const a = gT.row(t), b = gU.row(t); let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2; return [t, Math.sqrt(s)]; }));
  const maxMoved = Math.max(...moved.values());
  const evalWords = new Set(), answerWords = new Set();
  CASES.forEach(c => { tokenize(c.prompt).forEach(w => evalWords.add(w)); c.choices.forEach(w => evalWords.add(w)); answerWords.add(c.answer); });
  // Name each cluster by its three most frequent members
  const clusterNames = Array.from({ length: 8 }, (_, j) => ids.filter(t => clusters.get(t) === j).sort((a, b) => counts[b] - counts[a]).slice(0, 3).map(t => r.vocab[t]).join(', '));

  const wrap = $('#net-wrap'), svgEl = $('#net-svg');
  const W = wrap.clientWidth, H = wrap.clientHeight;
  const svg = d3.select(svgEl).attr('viewBox', [-W / 2, -H / 2, W, H]);
  svg.selectAll('*').remove();
  const root = svg.append('g');
  const zoom = d3.zoom().scaleExtent([0.15, 8]).on('zoom', e => root.attr('transform', e.transform));
  svg.call(zoom).on('dblclick.zoom', null);
  svg.on('click', e => { if (e.target === svgEl) selectWord(null); });
  const linkG = root.append('g'), nodeG = root.append('g');

  const prev = NET.graph && NET.graph.ri === ri ? NET.graph.positions() : null;
  const scale = Math.min(W, H) * 0.42;
  const pcaPos = stage => { const P = stage === 'trained' ? pcaT : pcaU; const xs = [...P.values()]; const mx = Math.max(...xs.map(p => Math.abs(p[0]))) || 1, my = Math.max(...xs.map(p => Math.abs(p[1]))) || 1; return t => [P.get(t)[0] / mx * scale * 1.25, P.get(t)[1] / my * scale]; };
  let pcaFn = pcaPos(S.stage);
  const nodes = ids.map(t => {
    const p = prev?.get(t) || pcaFn(t);
    return { id: t, w: r.vocab[t], x: p[0], y: p[1], r: 3 + 6 * Math.log1p(counts[t]) / (maxLog || 1) };
  });
  const byId = new Map(nodes.map(n => [n.id, n]));
  let stage = S.stage, links = [];

  function computeLinks() {
    const g = stage === 'trained' ? gT : gU, seen = new Set(), out = [];
    ids.forEach(t => neighbors(g, t, NET.k, allowed).forEach(([u, s]) => {
      if (s < NET.min) return; const key = t < u ? t + '-' + u : u + '-' + t; if (seen.has(key)) return; seen.add(key);
      out.push({ source: byId.get(t), target: byId.get(u), sim: s });
    }));
    return out;
  }
  function colorOf(n) {
    switch (NET.color) {
      case 'freq': return seqColor(Math.log1p(counts[n.id]) / (maxLog || 1) * 0.85 + 0.15);
      case 'moved': return seqColor(moved.get(n.id) / (maxMoved || 1) * 0.85 + 0.15);
      case 'eval': return answerWords.has(n.w) ? cssVar('--s2') : evalWords.has(n.w) ? cssVar('--s1') : cssVar('--line-2');
      default: return cssVar(`--s${clusters.get(n.id) + 1}`);
    }
  }
  function legend() {
    const L = $('#net-legend');
    if (NET.color === 'cluster') L.innerHTML = clusterNames.map((nm, j) => `<span><i class="swatch" style="background:var(--s${j + 1})"></i>${esc(nm)}</span>`).join('');
    else if (NET.color === 'eval') L.innerHTML = `<span><i class="swatch" style="background:var(--s2)"></i>an eval answer</span><span><i class="swatch" style="background:var(--s1)"></i>in an eval prompt or choice</span><span><i class="swatch" style="background:var(--line-2)"></i>not used by evals</span>`;
    else L.innerHTML = `<span>${NET.color === 'freq' ? 'rare' : 'barely moved'}</span><i class="swatch" style="width:90px;background:linear-gradient(90deg,${seqColor(.15)},${seqColor(1)})"></i><span>${NET.color === 'freq' ? 'frequent' : 'moved most'}</span>`;
  }
  let linkSel, nodeSel;
  const sim = d3.forceSimulation(nodes)
    .force('link', d3.forceLink().distance(d => 16 + (1 - d.sim) * 70).strength(d => 0.15 + 0.6 * Math.max(0, d.sim)))
    .force('charge', d3.forceManyBody().strength(-38).distanceMax(260))
    .force('collide', d3.forceCollide(d => d.r + 2))
    .on('tick', ticked);
  function applyLayout() {
    if (NET.layout === 'pca') { pcaFn = pcaPos(stage); sim.force('x', d3.forceX(d => pcaFn(d.id)[0]).strength(0.6)).force('y', d3.forceY(d => pcaFn(d.id)[1]).strength(0.6)); }
    else sim.force('x', d3.forceX(0).strength(0.035)).force('y', d3.forceY(0).strength(0.045));
  }
  function ticked() {
    linkSel.attr('x1', d => d.source.x).attr('y1', d => d.source.y).attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    nodeSel.attr('transform', d => `translate(${d.x},${d.y})`);
  }
  function drawLinks() {
    links = computeLinks();
    linkSel = linkG.selectAll('line').data(links, d => d.source.id + '-' + d.target.id)
      .join('line').attr('class', 'link').style('stroke-width', d => 0.4 + 2.2 * Math.max(0, d.sim) ** 2).style('opacity', d => 0.15 + 0.6 * Math.max(0, d.sim));
    sim.force('link').links(links);
    $('#net-stat').textContent = `${nodes.length} words · ${links.length} links · ${STAGES[stage].toLowerCase()} vectors`;
  }
  nodeSel = nodeG.selectAll('g.node').data(nodes, d => d.id).join(enter => {
    const g = enter.append('g').attr('class', 'node');
    g.append('circle').attr('r', d => d.r);
    g.append('text').attr('dx', d => d.r + 3).attr('dy', '0.35em').text(d => d.w);
    return g;
  });
  nodeSel.on('click', (e, d) => { e.stopPropagation(); selectWord(d.w); })
    .on('dblclick', (e, d) => { d.fx = null; d.fy = null; sim.alpha(0.3).restart(); })
    .on('mousemove', (e, d) => tip(e, `<b>${esc(d.w)}</b> · id ${d.id}<br>${num(counts[d.id])} training occurrences`)).on('mouseleave', untip)
    .call(d3.drag()
      .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.25).restart(); d.fx = d.x; d.fy = d.y; untip(); })
      .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
      .on('end', (e, d) => { if (!e.active) sim.alphaTarget(0); if (!NET.pinDrag) { d.fx = null; d.fy = null; } }));
  function recolor() { nodeSel.select('circle').style('fill', colorOf); legend(); }
  drawLinks(); recolor(); applyLayout(); sim.alpha(prev ? 0.5 : 1).restart();

  function highlight(word) {
    if (!word) { nodeSel.classed('dim', false).classed('sel', false); linkSel.classed('dim', false); return; }
    const t = r.vocab.indexOf(word), near = new Set([t]);
    links.forEach(l => { if (l.source.id === t) near.add(l.target.id); if (l.target.id === t) near.add(l.source.id); });
    nodeSel.classed('dim', d => !near.has(d.id)).classed('sel', d => d.id === t);
    linkSel.classed('dim', l => l.source.id !== t && l.target.id !== t);
  }
  NET.graph = {
    ri,
    positions: () => new Map(nodes.map(n => [n.id, [n.x, n.y]])),
    setStage(st) { stage = st; drawLinks(); applyLayout(); sim.alpha(0.8).restart(); highlight(NET.selected); renderInspector(); },
    update() { drawLinks(); sim.alpha(0.5).restart(); highlight(NET.selected); },
    recolor, relayout() { applyLayout(); sim.alpha(0.9).restart(); },
    highlight,
    centerOn(word) { const n = nodes.find(n => n.w === word); if (!n) return; svg.transition().duration(600).call(zoom.transform, d3.zoomIdentity.scale(1.8).translate(-n.x, -n.y)); },
    resetView() { nodes.forEach(n => { n.fx = null; n.fy = null; }); svg.transition().duration(400).call(zoom.transform, d3.zoomIdentity); sim.alpha(0.8).restart(); },
    clusters, clusterNames, counts, moved,
  };
  if (NET.selected && !r.vocab.includes(NET.selected)) NET.selected = null;
  highlight(NET.selected); renderInspector();
}

function selectWord(word, center) {
  NET.selected = word;
  if (NET.graph) { NET.graph.highlight(word); if (center && word) NET.graph.centerOn(word); }
  renderInspector();
}
function renderInspector() {
  const box = $('#net-inspector'); if (!box) return;
  const ri = S.run, r = RUNS[ri], word = NET.selected;
  if (!word) {
    const G = NET.graph;
    const movers = G ? [...G.moved.entries()].sort((a, b) => b[1] - a[1]).slice(0, 12).map(([t]) => r.vocab[t]) : [];
    box.innerHTML = `<div><span class="eyebrow">Inspector</span><h2 style="margin-top:4px">Pick a word</h2></div>
      <p class="note">Click any word in the network, or start with one of the words whose vectors changed most during training:</p>
      <div class="toks">${movers.map(w => `<span class="tok btn" data-w="${esc(w)}">${esc(w)}</span>`).join('')}</div>
      <p class="note">Try <b>hot</b>: its closest vectors are other describing words (words that fill the same slot in a sentence), not necessarily its opposite. The opposite shows up in <i>prediction</i> instead: run “the opposite of hot is” in the transformer lab.</p>`;
    $$('[data-w]', box).forEach(el => el.addEventListener('click', () => selectWord(el.dataset.w, true)));
    return;
  }
  const t = r.vocab.indexOf(word), gT = geometry(ri, 'trained'), gU = geometry(ri, 'untrained');
  const g = S.stage === 'trained' ? gT : gU, other = S.stage === 'trained' ? gU : gT;
  const nb = neighbors(g, t, 10, null).filter(([u]) => !/^<.+>$/.test(r.vocab[u]) || NET.special);
  const nbOther = neighbors(other, t, 6).map(([u]) => r.vocab[u]);
  const before = Array.from(gU.row(t)), after = Array.from(gT.row(t));
  const lo = Math.min(...before, ...after), hi = Math.max(...before, ...after);
  let cosBA = 0; for (let i = 0; i < before.length; i++) cosBA += gU.unit[t * gU.NE + i] * gT.unit[t * gT.NE + i];
  const m = model(ri, S.stage), out = m.forward([m.BOS, t]), p = m.softmax(out.logits, 1, true);
  const top = topK(p, 6, new Set([m.BOS]));
  const G = NET.graph;
  box.innerHTML = `
    <div><span class="eyebrow">Inspector · ${STAGES[S.stage]} vectors</span>
      <div class="row" style="margin-top:6px;align-items:baseline"><span class="word">${esc(word)}</span><span class="tok">id<sub></sub>${t}</span></div>
      <p class="note" style="margin-top:6px">${num(G?.counts[t])} occurrences in training text${G && G.clusters.has(t) ? ` · cluster <i class="swatch" style="background:var(--s${G.clusters.get(t) + 1})"></i> ${esc(G.clusterNames[G.clusters.get(t)])}` : ''}</p></div>
    <div class="stack" style="gap:6px">
      <span class="eyebrow">64-number vector, before → after training</span>
      <div id="strip-b"></div><div id="strip-a"></div>
      <p class="note">Blue = negative, red = positive; hover any cell for its value. Length ${fx(gU.norm[t], 3)} → ${fx(gT.norm[t], 3)}; direction kept: cosine(before, after) = ${fx(cosBA, 3)}.</p>
    </div>
    <div class="stack" style="gap:6px">
      <span class="eyebrow">Nearest words (cosine)</span>
      <div class="nb-list">${nb.map(([u, s]) => `<span class="tok btn" data-w="${esc(r.vocab[u])}">${esc(r.vocab[u])}</span><div class="bar-track"><div class="bar-fill" style="width:${Math.max(0, s) * 100}%"></div></div><span class="mono num" style="font-size:12px">${s.toFixed(3)}</span>`).join('')}</div>
      <p class="note">In the ${S.stage === 'trained' ? 'untrained' : 'trained'} model its neighbours were: ${nbOther.map(w => `<span class="mono">${esc(w)}</span>`).join(', ')}.</p>
    </div>
    <div class="stack" style="gap:6px">
      <span class="eyebrow">Right after “${esc(word)}”, the model predicts</span>
      ${probBars(top.map(i => ({ w: m.vocab[i], p: p[i] })))}
    </div>
    <div class="row"><button class="btn small" id="insp-lab">Open “${esc(word)}” in the transformer lab</button></div>`;
  const sb = stripCanvas(before, lo, hi), sa = stripCanvas(after, lo, hi);
  $('#strip-b', box).append(Object.assign(document.createElement('span'), { className: 'note', textContent: 'untrained' }), sb);
  $('#strip-a', box).append(Object.assign(document.createElement('span'), { className: 'note', textContent: 'trained' }), sa);
  $$('[data-w]', box).forEach(el => el.addEventListener('click', () => selectWord(el.dataset.w, true)));
  bindTips(box);
  $('#insp-lab', box).addEventListener('click', () => { LAB.prompt = word; go('lab'); });
}

/* ================================================================== TRANSFORMER LAB */
const LAB = { prompt: store.get('lab.prompt', 'the cup is not brown . it is'), head: 'mean', temp: 0.8, seed: 2026, greedy: false, note: '' };
const PRESETS = ['the customer', 'the report about the bus explains the', 'the opposite of hot is', 'the cup is not brown . it is', 'we learned about the local', 'a review of fruit helped us understand the'];
RENDER.lab = function () {
  const sec = $('#sec-lab'), r = run();
  sec.innerHTML = `
  <div class="sec-head"><div class="intro"><span class="eyebrow">Live forward pass · ${esc(r.label)} · ${STAGES[S.stage].toLowerCase()}</span><h1>Transformer lab</h1>
    <p>Type a prompt and follow it through the network: word → ID → vector, two attention blocks that mix in earlier words, then a probability for every word in the vocabulary. Nothing here updates the weights.</p></div></div>
  <div class="panel stack">
    <div class="prompt-row">
      <input id="lab-prompt" type="text" value="${esc(LAB.prompt)}" aria-label="Prompt" spellcheck="false">
      <button class="btn primary" id="lab-run">Run</button>
      <button class="btn" id="lab-step" title="Append the next word">Next word</button>
      <button class="btn" id="lab-cont" title="Append up to 12 words">Continue ×12</button>
    </div>
    <div class="chips">${PRESETS.map(p => `<button class="chip" data-p="${esc(p)}">${esc(p)}</button>`).join('')}</div>
    <div class="controls">
      <label class="ctl" for="lab-head">Attention shown <select id="lab-head"><option value="mean">Mean of 4 heads</option>${[0, 1, 2, 3].map(h => `<option value="${h}">Head ${h + 1}</option>`).join('')}</select></label>
      <label class="ctl" for="lab-temp">Temperature <input id="lab-temp" type="range" min="0.1" max="2" step="0.1" value="${LAB.temp}"><output id="lab-temp-o">${LAB.temp.toFixed(1)}</output></label>
      <label class="ctl" for="lab-seed">Seed <input id="lab-seed" type="number" value="${LAB.seed}" style="width:84px"></label>
      <label class="ctl"><input type="checkbox" id="lab-greedy" ${LAB.greedy ? 'checked' : ''}> Always pick the top word (greedy)</label>
    </div>
    <div id="lab-toks"></div>
  </div>
  <div class="grid g-side">
    <div class="panel" id="p-flow">
      <div class="panel-head"><h2>Information flow</h2><p>Each column is one position after that layer. Line thickness = attention weight. Drag boxes to untangle.</p></div>
      <div class="flow-wrap" id="flow"></div>
    </div>
    <div class="panel" id="p-next">
      <div class="panel-head"><h2>Next-word probabilities</h2></div>
      <div id="lab-next"></div>
    </div>
  </div>
  <div class="panel" id="p-heat">
    <div class="panel-head"><h2>Attention maps: 2 blocks × 4 heads</h2><p>Row = the word doing the looking; column = the earlier word it looks at. Each row sums to 1. The upper triangle is blank because the model cannot see future words.</p></div>
    <div id="lab-heat"></div>
  </div>
  <div class="panel" id="p-neurons">
    <div class="panel-head"><h2>Neuron map, block 2</h2><p>The 10 hidden MLP units that fire hardest at the last position, and the words each one pushes up most directly. Drag nodes to explore.</p></div>
    <div class="net-wrap" style="height:440px;min-height:0" id="neuron-wrap"><svg id="neuron-svg" aria-label="Neuron map"></svg></div>
    <p class="note" style="margin-top:8px">Approximation: a neuron's “direct effect” is its output weight vector passed through the final LayerNorm gain and compared with each word's embedding. It ignores the normalization scale and everything the neuron does through later computation.</p>
  </div>`;
  const inp = $('#lab-prompt');
  const runIt = () => { LAB.prompt = inp.value; store.set('lab.prompt', LAB.prompt); LAB.note = ''; labCompute(); };
  $('#lab-run').addEventListener('click', runIt);
  inp.addEventListener('keydown', e => { if (e.key === 'Enter') runIt(); });
  $$('[data-p]', sec).forEach(b => b.addEventListener('click', () => { inp.value = b.dataset.p; runIt(); }));
  $('#lab-head').value = LAB.head;
  $('#lab-head').addEventListener('change', e => { LAB.head = e.target.value; labCompute(); });
  $('#lab-temp').addEventListener('input', e => { LAB.temp = +e.target.value; $('#lab-temp-o').textContent = LAB.temp.toFixed(1); labCompute(); });
  $('#lab-seed').addEventListener('change', e => { LAB.seed = parseInt(e.target.value, 10) || 0; });
  $('#lab-greedy').addEventListener('change', e => { LAB.greedy = e.target.checked; });
  const extend = n => {
    const m = model(), enc = m.encode(inp.value), ids = [m.BOS, ...enc.ids];
    const g = m.generate(ids, { temperature: LAB.temp, maxTokens: n, seed: LAB.seed, greedy: LAB.greedy });
    LAB.seed += 1; $('#lab-seed').value = LAB.seed;
    inp.value = (inp.value.trim() + ' ' + g.text).trim();
    LAB.prompt = inp.value; store.set('lab.prompt', LAB.prompt);
    LAB.note = g.ended ? 'The model produced <code>&lt;EOS&gt;</code> (end of passage), so generation stopped.' : '';
    labCompute();
  };
  $('#lab-step').addEventListener('click', () => extend(1));
  $('#lab-cont').addEventListener('click', () => extend(12));
  labCompute();
};

function labCompute() {
  const m = model(), mo = model(S.run, S.stage === 'trained' ? 'untrained' : 'trained');
  const enc = m.encode(LAB.prompt), ids = [m.BOS, ...enc.ids];
  const out = m.forward(ids), p = m.softmax(out.logits, LAB.temp, true);
  const pOther = mo.softmax(mo.forward(ids).logits, LAB.temp, true);
  $('#lab-toks').innerHTML = `<span class="eyebrow">Tokens → IDs fed to the model</span><div style="margin-top:6px">${tokChips(m, out.ids, { unknownWords: enc.unknown })}</div>
    ${out.truncated ? `<p class="note" style="margin-top:6px">The prompt is longer than the ${m.BS}-token context window, so only the most recent ${m.BS} tokens are used.</p>` : ''}
    ${LAB.note ? `<p class="note" style="margin-top:6px">${LAB.note}</p>` : ''}`;
  const top = topK(p, 12, new Set([m.BOS]));
  const otherLabel = S.stage === 'trained' ? 'untrained model' : 'trained model';
  let ent = 0; for (let i = 0; i < p.length; i++) if (p[i] > 0) ent -= p[i] * Math.log2(p[i]);
  $('#lab-next').innerHTML = `
    <p class="note" style="margin-bottom:10px">Top word: <b class="mono">${esc(m.vocab[top[0]])}</b> at ${prob(p[top[0]])}. Spread (entropy) ${ent.toFixed(2)} bits; a uniform guess over ${m.V} words would be ${Math.log2(m.V).toFixed(2)} bits.</p>
    ${probBars(top.map(i => ({ w: m.vocab[i], p: p[i], ghost: pOther[i] })), { ghostLabel: otherLabel })}
    <p class="note" style="margin-top:10px">Bars: ${STAGES[S.stage].toLowerCase()} model at temperature ${LAB.temp.toFixed(1)}. Black tick: the ${otherLabel}, same prompt and temperature. Temperature divides the scores before softmax: below 1 sharpens, above 1 flattens. It changes sampling only, never the weights.</p>`;
  bindTips($('#lab-next'));
  drawFlow(m, out, p, top);
  drawHeat(m, out);
  drawNeurons(m, out);
}

function drawFlow(m, out, p, top) {
  const host = $('#flow'); host.innerHTML = '';
  const MAXT = 12, T = out.T, start = Math.max(0, T - MAXT), shown = T - start;
  const colX = [12, 214, 416, 618], rowH = 34, top0 = 38;
  const nOut = 8, W = 752, H = Math.max(shown, nOut) * rowH + top0 + 14;
  const nodes = [], links = [];
  const label = t => m.vocab[out.ids[t]];
  for (let l = 0; l <= m.NL; l++) for (let t = start; t < T; t++) nodes.push({ id: `L${l}_${t}`, l, t, x: colX[l], y: top0 + (t - start) * rowH, w: 104, text: label(t), cls: '' });
  const outIds = top.slice(0, nOut);
  outIds.forEach((v, i) => nodes.push({ id: `O_${v}`, l: 3, x: colX[3], y: top0 + i * rowH, w: 124, text: `${m.vocab[v]}  ${prob(p[v])}`, cls: 'out' }));
  const byId = new Map(nodes.map(n => [n.id, n]));
  for (let l = 1; l <= m.NL; l++) for (let t = start; t < T; t++) {
    links.push({ s: byId.get(`L${l - 1}_${t}`), d: byId.get(`L${l}_${t}`), w: 0, res: true });
    for (let s = start; s <= t; s++) {
      const heads = out.attn[l - 1];
      const w = LAB.head === 'mean' ? heads.reduce((a, h) => a + h[t][s], 0) / heads.length : heads[+LAB.head][t][s];
      if (w >= 0.05) links.push({ s: byId.get(`L${l - 1}_${s}`), d: byId.get(`L${l}_${t}`), w });
    }
  }
  const last = byId.get(`L${m.NL}_${T - 1}`);
  outIds.forEach(v => links.push({ s: last, d: byId.get(`O_${v}`), w: p[v], out: true }));

  const svg = d3.select(host).append('svg').attr('width', W).attr('height', H).attr('viewBox', [0, 0, W, H]);
  ['Tokens + position', 'After block 1', 'After block 2', 'Next word'].forEach((t, i) => svg.append('text').attr('class', 'col-label').attr('x', colX[i]).attr('y', 20).text(t));
  const path = d => { const x1 = d.s.x + d.s.w, y1 = d.s.y + 12, x2 = d.d.x, y2 = d.d.y + 12, mx = (x1 + x2) / 2; return `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`; };
  const linkSel = svg.append('g').selectAll('path').data(links).join('path').attr('d', path)
    .style('fill', 'none')
    .style('stroke', d => d.res ? 'var(--line-2)' : d.out ? 'var(--accent)' : 'var(--s1)')
    .style('stroke-dasharray', d => d.res ? '3 3' : null)
    .style('stroke-width', d => d.res ? 1 : 0.8 + d.w * 7)
    .style('opacity', d => d.res ? 0.8 : 0.25 + 0.75 * d.w)
    .on('mousemove', (e, d) => tip(e, d.res ? 'residual connection: each position keeps its own vector and adds the block\'s output' : d.out ? `P(<b>${esc(m.vocab[+d.d.id.slice(2)])}</b>) = ${prob(d.w)}` : `<b>${esc(d.d.text)}</b> (pos ${d.d.t}) attends to <b>${esc(d.s.text)}</b> (pos ${d.s.t}): ${(d.w * 100).toFixed(1)}%`))
    .on('mouseleave', untip);
  const nodeSel = svg.append('g').selectAll('g').data(nodes).join('g').attr('class', d => 'fnode ' + d.cls).attr('transform', d => `translate(${d.x},${d.y})`);
  nodeSel.append('rect').attr('width', d => d.w).attr('height', 24).attr('rx', 4);
  nodeSel.append('text').attr('x', 8).attr('y', 16).text(d => d.text.length > 17 ? d.text.slice(0, 16) + '…' : d.text);
  nodeSel.on('mouseenter', (e, n) => { nodeSel.classed('hl', d => d === n); linkSel.style('opacity', d => d.s === n || d.d === n ? 1 : d.res ? 0.3 : 0.08); })
    .on('mouseleave', () => { nodeSel.classed('hl', false); linkSel.style('opacity', d => d.res ? 0.8 : 0.25 + 0.75 * d.w); });
  nodeSel.call(d3.drag().on('drag', function (e, d) { d.x = Math.max(0, Math.min(W - d.w, d.x + e.dx)); d.y = Math.max(26, Math.min(H - 24, d.y + e.dy)); d3.select(this).attr('transform', `translate(${d.x},${d.y})`); linkSel.attr('d', path); }));
  if (start > 0) svg.append('text').attr('x', colX[0]).attr('y', H - 2).attr('class', 'col-label').text(`showing the last ${MAXT} of ${T} positions`);
}

function drawHeat(m, out) {
  const host = $('#lab-heat'), T = out.T, words = out.ids.map(i => m.vocab[i]);
  const showLabels = T <= 14, lab = showLabels ? 64 : 0, cell = Math.max(6, Math.min(26, Math.floor(220 / T)));
  const size = lab + T * cell;
  let html = '';
  for (let l = 0; l < m.NL; l++) {
    html += `<div class="eyebrow" style="margin:${l ? 16 : 0}px 0 8px">Block ${l + 1}</div><div class="heat-grid">`;
    for (let h = 0; h < m.NH; h++) {
      const A = out.attn[l][h];
      let cells = '';
      for (let t = 0; t < T; t++) for (let s = 0; s <= t; s++) {
        const w = A[t][s];
        cells += `<rect x="${lab + s * cell}" y="${lab + t * cell}" width="${cell - 1}" height="${cell - 1}" rx="1.5" style="fill:${seqColor(Math.sqrt(w))}" data-tip="${esc(`<b>${esc(words[t])}</b> → <b>${esc(words[s])}</b>: ${(w * 100).toFixed(1)}%`)}"/>`;
      }
      const labels = showLabels ? words.map((w, i) => `<text x="${lab - 4}" y="${lab + i * cell + cell / 2 + 3}" text-anchor="end" style="font:10px var(--f-mono);fill:var(--ink-2)">${esc(w.slice(0, 8))}</text><text transform="translate(${lab + i * cell + cell / 2 + 3},${lab - 4}) rotate(-60)" style="font:10px var(--f-mono);fill:var(--ink-2)">${esc(w.slice(0, 8))}</text>`).join('') : '';
      html += `<div class="heat"><span class="cap">head ${h + 1}</span><svg viewBox="0 0 ${size} ${size}" role="img" aria-label="Attention block ${l + 1} head ${h + 1}">${labels}${cells}</svg></div>`;
    }
    html += '</div>';
  }
  html += `<div class="legend" style="margin-top:10px"><span>0%</span><i class="swatch" style="width:120px;background:linear-gradient(90deg,${seqColor(0)},${seqColor(0.5)},${seqColor(1)})"></i><span>100% (colour uses √weight so small weights stay visible)</span></div>`;
  host.innerHTML = html;
  bindTips(host);
}

function drawNeurons(m, out) {
  const wrap = $('#neuron-wrap'), svgEl = $('#neuron-svg'); if (!wrap) return;
  const L = m.NL - 1, act = out.mlpLast[L], NE = m.NE, NF = m.NF;
  const cp = m.W[`transformer.h.${L}.mlp.c_proj.weight`], lnw = m.W['transformer.ln_f.weight'];
  const dirOf = j => { const d = new Float64Array(NE); for (let o = 0; o < NE; o++) d[o] = cp[o * NF + j] * lnw[o]; return d; };
  const scored = []; for (let j = 0; j < NF; j++) { const d = dirOf(j); scored.push([j, act[j] * Math.hypot(...d), d]); }
  scored.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const pick = scored.slice(0, 10), special = new Set([m.BOS, m.EOS, m.UNK]);
  const lastWord = m.vocab[out.ids[out.T - 1]];
  const nodes = [{ id: 'tok', type: 'tok', label: `“${lastWord}” (last position)` }], links = [], words = new Map();
  pick.forEach(([j, s, d]) => {
    nodes.push({ id: 'n' + j, type: 'neuron', label: `#${j}`, act: act[j] });
    links.push({ source: 'tok', target: 'n' + j, w: Math.abs(s), sign: Math.sign(act[j]) });
    const eff = []; for (let v = 0; v < m.V; v++) if (!special.has(v)) { let e = 0; for (let o = 0; o < NE; o++) e += d[o] * m.wte[v * NE + o]; eff.push([v, act[j] * e]); }
    eff.sort((a, b) => b[1] - a[1]);
    eff.slice(0, 3).forEach(([v, e]) => { if (!words.has(v)) { words.set(v, true); nodes.push({ id: 'w' + v, type: 'word', label: m.vocab[v] }); } links.push({ source: 'n' + j, target: 'w' + v, w: Math.max(0, e) }); });
  });
  const W = wrap.clientWidth || 800, H = wrap.clientHeight || 440;
  const svg = d3.select(svgEl).attr('viewBox', [-W / 2, -H / 2, W, H]); svg.selectAll('*').remove();
  const root = svg.append('g');
  svg.call(d3.zoom().scaleExtent([0.4, 4]).on('zoom', e => root.attr('transform', e.transform))).on('dblclick.zoom', null);
  const maxW = d3.max(links, l => l.w) || 1;
  const col = { tok: -W * 0.36, neuron: 0, word: W * 0.33 };
  const sim = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d => d.id).distance(90).strength(0.4))
    .force('charge', d3.forceManyBody().strength(-160))
    .force('x', d3.forceX(d => col[d.type]).strength(0.5)).force('y', d3.forceY(0).strength(0.06))
    .force('collide', d3.forceCollide(24));
  const link = root.append('g').selectAll('line').data(links).join('line')
    .style('stroke', d => d.source === 'tok' || d.source.id === 'tok' ? (d.sign < 0 ? 'var(--s8)' : 'var(--s1)') : 'var(--accent)')
    .style('stroke-width', d => 0.6 + d.w / maxW * 5).style('opacity', 0.55);
  const node = root.append('g').selectAll('g').data(nodes).join('g').attr('class', 'node').style('cursor', 'grab');
  node.append('circle').attr('r', d => d.type === 'tok' ? 11 : d.type === 'neuron' ? 8 : 6)
    .style('fill', d => d.type === 'tok' ? 'var(--ink)' : d.type === 'neuron' ? 'var(--s7)' : 'var(--accent)');
  node.append('text').attr('dx', d => d.type === 'tok' ? -14 : 11).attr('text-anchor', d => d.type === 'tok' ? 'end' : 'start').attr('dy', '0.35em').text(d => d.label);
  node.on('mousemove', (e, d) => tip(e, d.type === 'neuron' ? `hidden unit <b>${d.label}</b> in block 2<br>activation ${d.act.toFixed(3)}` : d.type === 'word' ? `<b>${esc(d.label)}</b>: pushed up directly by the neurons linked to it` : 'the last position of your prompt')).on('mouseleave', untip)
    .call(d3.drag().on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; }).on('end', (e) => { if (!e.active) sim.alphaTarget(0); }));
  sim.on('tick', () => { link.attr('x1', d => d.source.x).attr('y1', d => d.source.y).attr('x2', d => d.target.x).attr('y2', d => d.target.y); node.attr('transform', d => `translate(${d.x},${d.y})`); });
}

/* ================================================================== CHAT */
const CHAT = { turns: [], temp: 0.8, max: 24, seed: 2026, greedy: false };
RENDER.chat = function () {
  const sec = $('#sec-chat'), r = run(), m = model();
  const rec = DATA.chat_record;
  sec.innerHTML = `
  <div class="sec-head"><div class="intro"><span class="eyebrow">Chat interface · ${esc(r.label)} · ${STAGES[S.stage].toLowerCase()}</span><h1>Chat with the tiny model</h1>
    <p>This is a <b>tiny language model</b>. It continues your sentence one word at a time; it does not answer questions. Replies come from the saved weights of the run selected at the top, computed in this page.</p></div></div>
  <div class="panel stack" id="p-chat">
    <div class="label-banner"><span><b>${m.V}</b>-word vocabulary; other words become <code>&lt;UNK&gt;</code> (underlined red)</span><span><b>${m.BS}</b>-token context limit</span><span>Every prompt <b>starts fresh</b>: no memory between turns</span><span>Chatting never retrains the model</span></div>
    <div class="term" id="term" aria-live="polite"></div>
    <form class="chat-form" id="chat-form">
      <input id="chat-input" type="text" autocomplete="off" spellcheck="false" placeholder="Start a sentence, e.g. the patient" aria-label="Your prompt">
      <button class="btn primary" type="submit">Send</button>
    </form>
    <div class="controls">
      <label class="ctl" for="chat-temp">Temperature <input id="chat-temp" type="range" min="0.1" max="2" step="0.1" value="${CHAT.temp}"><output id="chat-temp-o">${CHAT.temp.toFixed(1)}</output></label>
      <label class="ctl" for="chat-max">Max words <input id="chat-max" type="number" min="1" max="48" value="${CHAT.max}" style="width:70px"></label>
      <label class="ctl" for="chat-seed">Next seed <input id="chat-seed" type="number" value="${CHAT.seed}" style="width:84px"></label>
      <label class="ctl"><input type="checkbox" id="chat-greedy"> Greedy</label>
      <button class="btn small" id="chat-copy" type="button">Copy transcript (JSON)</button>
      <button class="btn small" id="chat-clear" type="button">Clear browser turns</button>
    </div>
    <p class="note">Greedy replies match PyTorch exactly. Sampled replies use this page's own seeded random generator, so the same seed will not reproduce <code>chat.py</code> word for word. For graded evidence, use the recorded <code>chat.py</code> sessions below, which ran in PyTorch.</p>
  </div>
  <div class="panel" id="p-chat-records">
    <div class="panel-head"><h2>Recorded sessions from chat.py</h2><p>Real terminal sessions saved as JSON, with the model hash and settings.</p></div>
    ${[rec ? { src: 'results/chat_expanded.json', t: rec } : null, ...RUNS.map(rr => rr.chat ? { src: `llm_runs/${rr.id}/chat_transcript.json`, t: rr.chat } : null)].filter(Boolean).map(({ src, t }) => `
      <div class="stack" style="margin-bottom:16px"><div class="row">${fileLink(src)}<span class="faint" style="font-size:12px">${t.model ? 'model ' + esc(String(t.model).split('/').slice(-2).join('/')) : ''} ${t.model_sha256 ? '· sha ' + esc(t.model_sha256.slice(0, 10)) : ''} ${t.temperature != null ? '· T=' + t.temperature : ''}</span></div>
      <div class="scroll-x"><table><thead><tr><th>Prompt</th><th>Reply</th><th>Seed</th><th>Unknown words</th></tr></thead><tbody>${(t.turns || []).map(u => `<tr><td class="mono">${esc(u.prompt)}</td><td class="mono">${esc(u.response ?? u.reply ?? '')}</td><td class="num">${esc(u.seed ?? '')}</td><td class="mono">${esc((u.unknown_prompt_words || []).join(', '))}</td></tr>`).join('')}</tbody></table></div></div>`).join('') || '<p class="note">No recorded sessions yet. Run <code>python chat.py --model llm_runs/&lt;run&gt;/model.pt --transcript results/my-chat.json</code>.</p>'}
    ${CHECKS.screenshots.length ? `<p class="note">Screenshot evidence: ${CHECKS.screenshots.map(s => fileLink(s)).join(', ')}</p>` : ''}
  </div>`;
  const term = $('#term');
  const paint = () => {
    const seed = CHAT.turns.length ? '' : (rec?.turns || []).slice(0, 3).map(u => `<div class="you">${esc(u.prompt)}</div><div class="bot">${esc(u.response)}</div><div class="meta">recorded in PyTorch by chat.py · seed ${u.seed}</div>`).join('');
    term.innerHTML = seed + CHAT.turns.map(u => `<div class="you">${u.promptHtml}</div><div class="bot">${esc(u.reply) || '<i>(empty reply: the model ended the passage immediately)</i>'}</div><div class="meta">${esc(u.meta)}</div>`).join('');
    term.scrollTop = term.scrollHeight;
  };
  paint();
  $('#chat-temp').addEventListener('input', e => { CHAT.temp = +e.target.value; $('#chat-temp-o').textContent = CHAT.temp.toFixed(1); });
  $('#chat-max').addEventListener('change', e => { CHAT.max = Math.max(1, Math.min(48, parseInt(e.target.value, 10) || 24)); });
  $('#chat-seed').addEventListener('change', e => { CHAT.seed = parseInt(e.target.value, 10) || 0; });
  $('#chat-greedy').addEventListener('change', e => { CHAT.greedy = e.target.checked; });
  $('#chat-form').addEventListener('submit', e => {
    e.preventDefault();
    const text = $('#chat-input').value.trim(); if (!text) return;
    const mm = model(), enc = mm.encode(text), ids = [mm.BOS, ...enc.ids];
    const g = mm.generate(ids, { temperature: CHAT.temp, maxTokens: CHAT.max, seed: CHAT.seed, greedy: CHAT.greedy });
    const unk = new Set(enc.unknown);
    CHAT.turns.push({
      prompt: text, reply: g.text, seed: CHAT.seed, unknown: enc.unknown, truncated: ids.length > mm.BS, run: run().id, stage: S.stage,
      promptHtml: enc.toks.map(t => unk.has(t) ? `<span class="unkw" title="not in vocabulary → <UNK>">${esc(t)}</span>` : esc(t)).join(' '),
      meta: `browser · ${run().label} · ${S.stage} · ${CHAT.greedy ? 'greedy' : 'T=' + CHAT.temp.toFixed(1) + ' seed ' + CHAT.seed}${enc.unknown.length ? ' · unknown: ' + enc.unknown.join(', ') : ''}${ids.length > mm.BS ? ' · prompt cut to last ' + mm.BS + ' tokens' : ''}${g.ended ? ' · stopped at <EOS>' : ' · hit word limit'}`,
    });
    CHAT.seed += 1; $('#chat-seed').value = CHAT.seed; $('#chat-input').value = ''; paint();
  });
  $('#chat-clear').addEventListener('click', () => { CHAT.turns = []; paint(); });
  $('#chat-copy').addEventListener('click', e => {
    const json = JSON.stringify({ interface: 'nanoGPT Lab Bench (browser)', run: run().id, stage: S.stage, fresh_context_per_prompt: true, turns: CHAT.turns.map(({ promptHtml, meta, ...u }) => u) }, null, 2);
    const btn = e.currentTarget;
    const done = ok => { btn.textContent = ok ? 'Copied' : 'Copy failed: select the text in the log'; setTimeout(() => { btn.textContent = 'Copy transcript (JSON)'; }, 1800); };
    try { navigator.clipboard.writeText(json).then(() => done(true), () => done(false)); } catch (err) { done(false); }
  });
};

/* ================================================================== TRAINING */
RENDER.training = function () {
  const sec = $('#sec-training'), r = run(), c = r.config, ins = r.inspection, m = model(S.run, 'trained');
  const steps = Object.keys(r.samples).map(Number).sort((a, b) => a - b);
  const temps = r.temperature ? Object.keys(r.temperature) : [];
  const fu = ins.first_update;
  const delta = fu ? fu.before - fu.after : null;
  const pb = ins.probabilities_before, pa = ins.probabilities_after;
  const topAfter = pa ? topK(pa, 10) : [];
  sec.innerHTML = `
  <div class="sec-head"><div class="intro"><span class="eyebrow">Evidence · ${esc(r.label)}</span><h1>Training evidence</h1>
    <p>What the saved files show about this run: settings, losses, samples at each checkpoint, one token followed from text to vector, and one real gradient step.</p></div></div>
  <div class="stats">
    <div class="stat"><span class="v">${num(r.summary?.completed_steps)}</span><span class="k">steps completed${r.summary?.interrupted ? ' (interrupted)' : ''}</span></div>
    <div class="stat"><span class="v">${fx(r.summary?.elapsed_seconds, 1)}<small> s</small></span><span class="k">training time, CPU</span></div>
    <div class="stat"><span class="v">${num(c.parameters)}</span><span class="k">parameters</span></div>
    <div class="stat"><span class="v">${c.learning_rate}</span><span class="k">peak learning rate (warmup + cosine)</span></div>
    <div class="stat"><span class="v">${num(c.train_documents)}<small> / ${num(c.validation_documents)}</small></span><span class="k">train / validation passages</span></div>
    <div class="stat"><span class="v">${pct(c.training_unknown_rate, 2)}<small> / ${pct(c.validation_unknown_rate, 2)}</small></span><span class="k">unknown-token rate, train / held-out</span></div>
  </div>

  <div class="grid g2">
    <div class="panel" id="p-loss">
      <div class="panel-head"><h2>Loss curves, all runs</h2></div>
      <div class="legend" id="loss-legend"></div>
      <div class="chart" id="loss-chart" style="margin-top:8px"></div>
      <p class="note" style="margin-top:8px">Measured on fixed panels of ${c.evaluation_panel_size?.train ?? 20} training and ${c.evaluation_panel_size?.validation ?? 20} validation passages (mean over non-padding next-token targets). These are small estimates, not full-corpus losses, and corpora differ, so compare curves within a run, not across runs.</p>
    </div>
    <div class="panel">
      <div class="panel-head"><h2>Every measured loss</h2><p>From <code>history.json</code></p></div>
      <div class="scroll-x"><table><thead><tr><th>Run</th><th class="n">Step</th><th class="n">Training panel</th><th class="n">Validation panel</th><th class="n">Gap</th></tr></thead><tbody>
      ${RUNS.map((rr, i) => rr.history.map(h => `<tr><td><span class="swatch" style="background:${runColor(i)}"></span> ${esc(rr.label)}</td><td class="n">${num(h.step)}</td><td class="n">${fx(h.training_loss, 4)}</td><td class="n">${fx(h.validation_loss, 4)}</td><td class="n">${fx(h.validation_loss - h.training_loss, 4)}</td></tr>`).join('')).join('')}
      </tbody></table></div>
      <p class="note" style="margin-top:8px">Files: ${fileLink(`llm_runs/${r.id}/history.json`, 'history.json')} · ${fileLink(`llm_runs/${r.id}/training.csv`, 'training.csv')} · ${fileLink(`llm_runs/${r.id}/training_curves.svg`, 'training_curves.svg')} · ${fileLink(`llm_runs/${r.id}/training_summary.json`, 'training_summary.json')} · ${fileLink(`llm_runs/${r.id}/config.json`, 'config.json')}</p>
    </div>
  </div>

  <div class="panel" id="p-samples">
    <div class="panel-head"><h2>Samples: untrained, halfway, final</h2><p>Same generation settings at every checkpoint. Shown exactly as saved, including empty or garbled lines.</p></div>
    <div class="samples">${steps.map((s, i) => `<div class="sample-col"><span class="eyebrow">Step ${num(s)} · ${i === 0 ? 'untrained' : i === steps.length - 1 ? 'final' : 'halfway'}</span>
      <ol>${r.samples[s].map(l => `<li>${l.trim() ? esc(l) : '<span class="empty-sample">(empty)</span>'}</li>`).join('')}</ol>${fileLink(`llm_runs/${r.id}/samples/step_${String(s).padStart(4, '0')}.txt`)}</div>`).join('')}</div>
  </div>

  <div class="grid g2">
    <div class="panel" id="p-inspect-token">
      <div class="panel-head"><h2>Text → tokens → IDs</h2><p>From <code>tokenization.json</code></p></div>
      ${r.tokenization ? `<p class="mono" style="font-size:13px">“${esc(r.tokenization.example)}”</p>
      <div style="margin-top:10px">${tokChips(m, r.tokenization.ids)}</div>
      <p class="note" style="margin-top:10px">Training pairs every position with the next one. Input IDs <code>${esc(r.tokenization.inputs.slice(0, 6).join(', '))}…</code> must predict targets <code>${esc(r.tokenization.targets.slice(0, 6).join(', '))}…</code>, the same sequence shifted by one.</p>` : '<p class="note">No tokenization.json in this run.</p>'}
      <div class="stack" style="margin-top:14px;gap:6px">
        <span class="eyebrow">Word “${esc(ins.token)}” → ID ${ins.token_id} → 64 numbers</span>
        <div id="ins-before"></div><div id="ins-after"></div>
        <details><summary class="note">Show all 64 values, before and after</summary>
          <div class="scroll-x" style="max-height:260px;overflow-y:auto;margin-top:6px"><table><thead><tr><th class="n">dim</th><th class="n">before</th><th class="n">after</th><th class="n">change</th></tr></thead><tbody>
          ${ins.embedding_before.map((b, i) => `<tr><td class="n">${i}</td><td class="n mono">${b.toFixed(5)}</td><td class="n mono">${ins.embedding_after[i].toFixed(5)}</td><td class="n mono">${(ins.embedding_after[i] - b).toFixed(5)}</td></tr>`).join('')}
          </tbody></table></div></details>
      </div>
    </div>
    <div class="panel" id="p-update">
      <div class="panel-head"><h2>One real gradient and weight update</h2><p>From <code>inspection.json</code></p></div>
      ${fu ? `<p class="note">The very first optimizer step changed coordinate <b>${fu.coordinate}</b> of the <b class="mono">${esc(fu.token)}</b> embedding:</p>
      <div class="stats" style="margin-top:10px;grid-template-columns:repeat(2,minmax(0,1fr))">
        <div class="stat"><span class="v mono" style="font-size:18px">${sci(fu.before)}</span><span class="k">weight before</span></div>
        <div class="stat"><span class="v mono" style="font-size:18px">${sci(fu.gradient)}</span><span class="k">gradient ∂loss/∂weight</span></div>
        <div class="stat"><span class="v mono" style="font-size:18px">${sci(fu.learning_rate)}</span><span class="k">learning rate at step 1 (warmup)</span></div>
        <div class="stat"><span class="v mono" style="font-size:18px">${sci(fu.after)}</span><span class="k">weight after</span></div>
      </div>
      <p class="note" style="margin-top:10px">Change = ${sci(-delta)}. The gradient is positive, so raising this weight would raise the loss; the optimizer lowered it. AdamW's first step moves each weight by about <i>learning rate × sign(gradient)</i> = ${sci(fu.learning_rate)}, plus weight decay of lr × 0.01 × weight = ${sci(fu.learning_rate * 0.01 * fu.before)}. Predicted change ${sci(fu.learning_rate * (1 + 0.01 * fu.before))}; measured ${sci(delta)}.</p>` : '<p class="note">No first-update record.</p>'}
      <div class="stack" style="margin-top:14px;gap:6px" id="p-probs">
        <span class="eyebrow">Next-word probabilities after “${esc(ins.prefix)}”, before vs after</span>
        ${pa ? probBars(topAfter.map(i => ({ w: r.vocab[i], p: pa[i], ghost: pb[i] })), { ghostLabel: 'before training' }) : ''}
        <p class="note">Bars = trained model; tick = untrained (about 1/${r.vocab.length} = ${prob(1 / r.vocab.length)} each, close to uniform).</p>
      </div>
    </div>
  </div>

  <div class="grid g2">
    <div class="panel" id="p-attn">
      <div class="panel-head"><h2>Saved attention rows</h2><p>Block 1, head 1, prefix “${esc(ins.prefix)}”</p></div>
      ${ins.attention_rows ? attnTable(['<BOS>', ...tokenize(ins.prefix)], ins.attention_rows) : ''}
      <p class="note" style="margin-top:8px">Each row is one position deciding how much to read from itself and earlier positions. The lab page shows every block and head for any prompt.</p>
    </div>
    <div class="panel" id="p-temp">
      <div class="panel-head"><h2>Temperature comparison</h2><p>Same trained weights, starting token and seeds; only the sampling temperature changes.</p></div>
      ${temps.length ? `<div class="scroll-x"><table><thead><tr><th>#</th>${temps.map(t => `<th>T = ${esc(t)}</th>`).join('')}</tr></thead><tbody>
        ${r.temperature[temps[0]].map((_, i) => `<tr><td class="num">${i + 1}</td>${temps.map(t => `<td class="mono" style="font-size:12px">${esc(r.temperature[t][i]) || '<span class="empty-sample">(empty)</span>'}</td>`).join('')}</tr>`).join('')}
      </tbody></table></div>
      <p class="note" style="margin-top:8px">${temps.every(t => JSON.stringify(r.temperature[t]) === JSON.stringify(r.temperature[temps[0]])) ? 'All three temperatures produced identical text here. The trained model is so confident on this synthetic corpus that rescaling the scores rarely changes which word is drawn. Try the temperature slider in the lab on a less predictable prompt to see the effect.' : 'Low temperature repeats the most likely words; high temperature takes more risks.'}</p>
      ${fileLink(`llm_runs/${r.id}/temperature_comparison.json`)}` : '<p class="note">No temperature comparison saved.</p>'}
    </div>
  </div>`;
  $('#ins-before', sec).append(Object.assign(document.createElement('span'), { className: 'note', textContent: 'before training (random init)' }), stripCanvas(ins.embedding_before, Math.min(...ins.embedding_before, ...ins.embedding_after), Math.max(...ins.embedding_before, ...ins.embedding_after)));
  $('#ins-after', sec).append(Object.assign(document.createElement('span'), { className: 'note', textContent: 'after training' }), stripCanvas(ins.embedding_after, Math.min(...ins.embedding_before, ...ins.embedding_after), Math.max(...ins.embedding_before, ...ins.embedding_after)));
  lossChart();
  bindTips(sec);
};
function attnTable(words, rows) {
  return `<div class="scroll-x"><table><thead><tr><th>query ↓ · key →</th>${words.map(w => `<th class="mono">${esc(w)}</th>`).join('')}</tr></thead><tbody>${rows.map((row, t) => `<tr><td class="mono">${esc(words[t])}</td>${row.map((w, s) => s > t ? '<td></td>' : `<td><div class="heatcell" style="background:${seqColor(w)};color:${inkOn(seqColor(w))}">${(w * 100).toFixed(1)}%</div></td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}
function lossChart() {
  const host = $('#loss-chart'); if (!host) return;
  const W = 560, H = 280, M = { l: 44, r: 16, t: 10, b: 34 };
  const series = RUNS.flatMap((r, i) => ['training_loss', 'validation_loss'].map(k => ({ r, i, k, pts: r.history.map(h => ({ x: h.step, y: h[k] })) })));
  const x = d3.scaleLinear().domain([0, d3.max(series, s => d3.max(s.pts, p => p.x)) || 1]).range([M.l, W - M.r]);
  const y = d3.scaleLinear().domain([0, (d3.max(series, s => d3.max(s.pts, p => p.y)) || 1) * 1.05]).nice().range([H - M.b, M.t]);
  const svg = d3.select(host).append('svg').attr('viewBox', [0, 0, W, H]).attr('role', 'img').attr('aria-label', 'Loss by training step for each run');
  svg.append('g').attr('class', 'grid').selectAll('line').data(y.ticks(5)).join('line').attr('x1', M.l).attr('x2', W - M.r).attr('y1', d => y(d)).attr('y2', d => y(d));
  svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${H - M.b})`).call(d3.axisBottom(x).ticks(6).tickSizeOuter(0));
  svg.append('g').attr('class', 'axis').attr('transform', `translate(${M.l},0)`).call(d3.axisLeft(y).ticks(5).tickSizeOuter(0));
  svg.append('text').attr('x', W - M.r).attr('y', H - 4).attr('text-anchor', 'end').style('font', '11px var(--f-body)').style('fill', 'var(--ink-3)').text('training step');
  svg.append('text').attr('x', M.l).attr('y', M.t + 2).attr('dx', 6).style('font', '11px var(--f-body)').style('fill', 'var(--ink-3)').text('cross-entropy loss');
  const line = d3.line().x(p => x(p.x)).y(p => y(p.y));
  series.forEach(s => {
    svg.append('path').attr('d', line(s.pts)).style('fill', 'none').style('stroke', runColor(s.i)).style('stroke-width', 2).style('stroke-dasharray', s.k === 'validation_loss' ? '5 4' : null);
    svg.append('g').selectAll('circle').data(s.pts).join('circle').attr('cx', p => x(p.x)).attr('cy', p => y(p.y)).attr('r', 4.5)
      .style('fill', s.k === 'validation_loss' ? 'var(--panel)' : runColor(s.i)).style('stroke', runColor(s.i)).style('stroke-width', 2)
      .on('mousemove', (e, p) => tip(e, `${esc(s.r.label)}<br>${s.k === 'training_loss' ? 'training' : 'validation'} panel · step ${num(p.x)}<br><b>${p.y.toFixed(4)}</b>`)).on('mouseleave', untip);
  });
  $('#loss-legend').innerHTML = RUNS.map((r, i) => `<span><i class="swatch" style="background:${runColor(i)}"></i>${esc(r.label)}</span>`).join('') + `<span>── training panel</span><span>╌╌ validation panel</span>`;
}

/* ================================================================== EVALS */
const EV = { group: 'all', cat: 'all', outcome: 'all', q: '', open: null };
const CAT_ORDER = ['domain_context', 'domain_place', 'new_wording', 'grammar', 'opposites', 'negation', 'reference', 'sequence', 'spatial_relations', 'everyday_knowledge', 'categories_and_analogies'];
const COLS = () => RUNS.flatMap((r, i) => ['untrained', 'final'].filter(s => r.evals[s]).map(s => ({ r, i, s, res: Object.fromEntries(r.evals[s].results.map(x => [x.id, x])), sum: r.evals[s].summary })));
RENDER.evals = function () {
  const sec = $('#sec-evals'), cols = COLS();
  const cats = [...new Set([...CAT_ORDER, ...CASES.map(c => c.category)])].filter(c => CASES.some(x => x.category === c));
  const groups = [...new Set(CASES.map(c => c.group))];
  const leak = CHECKS.leakage;
  sec.innerHTML = `
  <div class="sec-head"><div class="intro"><span class="eyebrow">48 fixed cases · ${esc(DATA.suite.id)}</span><h1>Language evals</h1>
    <p>A case scores 1 only when the correct word gets the highest probability among its four choices. Ties score 0. A case with a word the model has never seen is unscorable and counts as 0. These public cases guided the corpus changes, so treat them as a development benchmark, not an unseen test.</p></div></div>
  <div class="panel" id="p-four-row">
    <div class="panel-head"><h2>Comparison: every experiment, before and after training</h2></div>
    <div class="scroll-x"><table><thead><tr><th>Experiment</th><th>Stage</th><th>All-case success</th><th class="n">Correct</th><th class="n">Scorable</th><th class="n">Accuracy on scorable</th><th class="n">Vocabulary coverage</th><th>Full results</th></tr></thead><tbody>
    ${cols.map(c => { const o = c.sum.overall; return `<tr><td><span class="swatch" style="background:${runColor(c.i)}"></span> ${esc(c.r.label)}</td><td>${c.s === 'final' ? 'Trained' : 'Untrained'}</td>
      <td><div class="bar-track" style="width:140px"><div class="bar-fill" style="width:${o.success_rate_all_cases * 100}%;background:${runColor(c.i)};opacity:${c.s === 'final' ? 1 : 0.45}"></div></div></td>
      <td class="n"><b>${o.correct}</b> / ${o.total} · ${pct(o.success_rate_all_cases)}</td><td class="n">${o.scorable}</td><td class="n">${pct(o.accuracy_scorable_cases)}</td><td class="n">${pct(o.coverage)}</td>
      <td>${Object.entries(c.r.evals[c.s].files).map(([k, p]) => fileLink(p, k.toUpperCase())).join(' · ')}</td></tr>`; }).join('')}
    </tbody></table></div>
  </div>
  <div class="panel" id="p-cats">
    <div class="panel-head"><h2>By category</h2><p>Correct / total. The small line is vocabulary coverage: the share of cases whose words the model knows. Low coverage means missing vocabulary, not a failure to learn a pattern.</p></div>
    <div class="scroll-x"><table><thead><tr><th>Category</th><th>Group</th>${cols.map(c => `<th style="text-align:center"><span class="swatch" style="background:${runColor(c.i)}"></span> ${esc(c.r.label.split(':')[0])}<br><span class="faint">${c.s === 'final' ? 'trained' : 'untrained'}</span></th>`).join('')}</tr></thead><tbody>
    ${cats.map(cat => { const g = CASES.find(x => x.category === cat).group; return `<tr><td><b>${esc(cat.replace(/_/g, ' '))}</b></td><td class="faint" style="font-size:12px">${esc(g.replace(/_/g, ' '))}</td>${cols.map(c => { const b = c.sum.by_category[cat]; if (!b) return '<td></td>'; const bg = seqColor(b.success_rate_all_cases * 0.9 + (b.correct ? 0.1 : 0)); return `<td><div class="heatcell" style="background:${bg};color:${inkOn(bg)}">${b.correct}/${b.total}<small>cov ${pct(b.coverage, 0)}</small></div></td>`; }).join('')}</tr>`; }).join('')}
    </tbody></table></div>
  </div>
  <div class="panel" id="p-cases">
    <div class="panel-head"><h2>All 48 cases</h2><p>Click a row for choice probabilities in every run and the free continuation.</p></div>
    <div class="controls" style="margin-bottom:12px">
      <label class="ctl" for="ev-q">Search <input id="ev-q" type="search" placeholder="prompt, answer, id…" value="${esc(EV.q)}" style="width:180px"></label>
      <label class="ctl" for="ev-group">Group <select id="ev-group"><option value="all">All</option>${groups.map(g => `<option value="${g}">${esc(g.replace(/_/g, ' '))}</option>`).join('')}</select></label>
      <label class="ctl" for="ev-cat">Category <select id="ev-cat"><option value="all">All</option>${cats.map(g => `<option value="${g}">${esc(g.replace(/_/g, ' '))}</option>`).join('')}</select></label>
      <label class="ctl" for="ev-out">Latest trained run <select id="ev-out"><option value="all">Any outcome</option><option value="pass">Passes</option><option value="fail">Fails (scorable)</option><option value="unk">Unscorable</option><option value="improved">Improved vs first run</option></select></label>
      <span class="legend"><span><i class="mark pass">✓</i> correct</span><span><i class="mark fail">✗</i> wrong</span><span><i class="mark unk">?</i> unknown word</span></span>
    </div>
    <div class="scroll-x"><table id="ev-table"><thead><tr><th>ID</th><th>Category</th><th>Prompt → answer</th>${cols.map(c => `<th style="text-align:center" title="${esc(c.r.label)} ${c.s}"><span class="swatch" style="background:${runColor(c.i)}"></span><br>${c.s === 'final' ? 'T' : 'U'}</th>`).join('')}</tr></thead><tbody></tbody></table></div>
  </div>
  <div class="grid g2">
    <div class="panel" id="p-probes">
      <div class="panel-head"><h2>Control probes: coverage or learned pattern?</h2></div>
      ${probesHTML()}
    </div>
    <div class="panel" id="p-leak">
      <div class="panel-head"><h2>Keeping the evals out of training</h2></div>
      <div class="stack">
        <div class="row"><span class="pill ${leak.hits.length ? 'todo' : 'done'}">${leak.hits.length ? leak.hits.length + ' matches found' : 'No eval prompt found in training text'}</span></div>
        <p class="note">Re-checked when this dashboard was built: all ${leak.prompts} eval prompts were tokenized and searched as contiguous word sequences in ${leak.sources_checked.length} training inputs: ${leak.sources_checked.map(s => `<code>${esc(s)}</code>`).join(', ')}.</p>
        ${leak.hits.length ? `<div class="callout warn">${leak.hits.map(h => `${esc(h.case)} in ${esc(h.source)}`).join('<br>')}</div>` : ''}
        ${RUNS.map(r => r.eval_separation ? `<p class="note"><b>${esc(r.label)}</b>: the notebook withheld ${r.eval_separation.excluded_passages} generated classroom passages that contained a reserved test prefix (${r.eval_separation.case_ids.length} cases), before splitting or building the vocabulary. ${fileLink(`llm_runs/${r.id}/eval_separation.json`, 'eval_separation.json')}</p>` : '').join('')}
        <p class="note">Suite file SHA-256 <code>${esc(CHECKS.suite_sha256.slice(0, 16))}…</code> ${CHECKS.suite_sha256 === CHECKS.suite_sha256_expected ? '<span class="pill done">unchanged</span>' : '<span class="pill todo">changed since first copy</span>'}</p>
        <p class="note">Neither check catches paraphrases. The extension text was written with different people, objects and sentence frames from the eval stories.</p>
      </div>
    </div>
  </div>
  <div class="panel" id="p-rerun">
    <div class="panel-head"><h2>Rerun the evals on saved weights</h2></div>
    <pre class="mono" style="margin:0;white-space:pre-wrap;font-size:12.5px;background:var(--sunken);padding:12px;border-radius:6px">${RUNS.map(r => `python run_evals.py --model llm_runs/${r.id}/model.pt --output results/rerun-${r.id.slice(0, 8)}-final\npython run_evals.py --model llm_runs/${r.id}/model_untrained.pt --stage untrained --output results/rerun-${r.id.slice(0, 8)}-untrained`).join('\n')}</pre>
    <p class="note" style="margin-top:8px">The runner sends only the prompt to the model, never the choices or answer key, and never updates weights. The output folder must not exist yet.</p>
  </div>`;
  $('#ev-group').value = EV.group; $('#ev-cat').value = EV.cat; $('#ev-out').value = EV.outcome;
  $('#ev-q').addEventListener('input', e => { EV.q = e.target.value; caseRows(); });
  $('#ev-group').addEventListener('change', e => { EV.group = e.target.value; caseRows(); });
  $('#ev-cat').addEventListener('change', e => { EV.cat = e.target.value; caseRows(); });
  $('#ev-out').addEventListener('change', e => { EV.outcome = e.target.value; caseRows(); });
  caseRows();
};
function caseMark(x) {
  if (!x) return '<span class="faint">·</span>';
  if (x.status !== 'scored') return `<span class="mark unk" title="${esc('unscorable: ' + [...(x.unknown_prompt_words || []), ...(x.unknown_choices || [])].join(', '))}">?</span>`;
  return x.score ? '<span class="mark pass">✓</span>' : `<span class="mark fail" title="picked ${esc(x.predicted_choice)}">✗</span>`;
}
function caseRows() {
  const cols = COLS(), tb = $('#ev-table tbody'); if (!tb) return;
  const lastT = [...cols].reverse().find(c => c.s === 'final'), firstT = cols.find(c => c.s === 'final');
  const q = EV.q.trim().toLowerCase();
  const rows = CASES.filter(c => (EV.group === 'all' || c.group === EV.group) && (EV.cat === 'all' || c.category === EV.cat)
    && (!q || (c.id + ' ' + c.prompt + ' ' + c.answer + ' ' + c.choices.join(' ') + ' ' + c.category).toLowerCase().includes(q))
    && (() => { if (EV.outcome === 'all' || !lastT) return true; const x = lastT.res[c.id];
      if (EV.outcome === 'pass') return x?.score === 1; if (EV.outcome === 'fail') return x?.status === 'scored' && !x.score; if (EV.outcome === 'unk') return x?.status !== 'scored';
      if (EV.outcome === 'improved') return x?.score === 1 && firstT?.res[c.id]?.score !== 1; return true; })());
  tb.innerHTML = rows.map(c => `<tr class="clickable${EV.open === c.id ? ' open' : ''}" data-case="${c.id}" id="case-${c.id}"><td class="mono">${c.id}</td><td style="font-size:12px">${esc(c.category.replace(/_/g, ' '))}</td>
    <td><span class="mono" style="font-size:12.5px">${esc(c.prompt)} <b style="color:var(--accent)">${esc(c.answer)}</b></span></td>${cols.map(col => `<td style="text-align:center">${caseMark(col.res[c.id])}</td>`).join('')}</tr>
    ${EV.open === c.id ? `<tr class="open"><td colspan="${3 + cols.length}">${caseDetail(c, cols)}</td></tr>` : ''}`).join('') || `<tr><td colspan="${3 + cols.length}" class="faint">No cases match these filters.</td></tr>`;
  $$('tr[data-case]', tb).forEach(tr => tr.addEventListener('click', () => { EV.open = EV.open === tr.dataset.case ? null : tr.dataset.case; caseRows(); }));
  $$('[data-trylab]', tb).forEach(b => b.addEventListener('click', e => { e.stopPropagation(); LAB.prompt = b.dataset.trylab; go('lab'); }));
  bindTips(tb);
}
function caseDetail(c, cols) {
  return `<div class="stack" style="padding:6px 0">
    <p class="note"><b>Why:</b> ${esc(c.reason || '')}</p>
    <div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(220px,1fr))">${cols.map(col => { const x = col.res[c.id]; if (!x) return ''; const cp = x.choice_probabilities || {};
      return `<div class="stack" style="gap:6px"><span class="eyebrow"><span class="swatch" style="background:${runColor(col.i)}"></span> ${esc(col.r.label.split(':')[0])} · ${col.s === 'final' ? 'trained' : 'untrained'} ${caseMark(x)}</span>
        ${x.status === 'scored' ? probBars(c.choices.map(w => ({ w: w + (w === c.answer ? ' ★' : ''), p: cp[w] ?? 0, color: w === c.answer ? 'var(--good)' : 'var(--line-2)' }))) : `<p class="note">Unscorable. Unknown: ${esc([...(x.unknown_prompt_words || []), ...(x.unknown_choices || [])].join(', '))}</p>`}
        <p class="note">Free continuation: <span class="mono">${esc(x.generated_text) || '<i>(empty)</i>'}</span></p></div>`; }).join('')}</div>
    <div><button class="btn small" data-trylab="${esc(c.prompt)}">Trace this prompt in the transformer lab</button></div></div>`;
}
function probesHTML() {
  const P = DATA.probes; if (!P || !Object.keys(P).length) return '<p class="note">No control probes saved.</p>';
  let h = '';
  if (P.opposites?.rows) {
    const rows = P.opposites.rows, first = rows.filter(r => r.partner_rank === 1).length;
    h += `<p class="note">Opposites (expanded run): for each adjective stem, where does its true antonym rank among ${P.opposites.candidates?.length ?? '?'} candidates? <b>${first} of ${rows.length}</b> rank first.</p>
    <div class="scroll-x" style="max-height:220px;overflow-y:auto;margin:8px 0 14px"><table><thead><tr><th>Stem</th><th>Antonym</th><th class="n">Rank</th><th>Model's top pick</th></tr></thead><tbody>${rows.map(r => `<tr><td class="mono">${esc(r.stem)}</td><td class="mono">${esc(r.partner)}</td><td class="n">${r.partner_rank === 1 ? '<span class="mark pass">1</span>' : r.partner_rank}</td><td class="mono">${esc(r.top)}</td></tr>`).join('')}</tbody></table></div>`;
  }
  if (P.negation) {
    const n = P.negation, rows = n.rows || n.probes || [];
    h += `<p class="note">Negation: ${esc(n.summary || n.note || 'probes that swap the story while keeping the frame; a model that learned negation should change its answer with the story.')}</p>
    ${rows.length ? `<div class="scroll-x" style="max-height:220px;overflow-y:auto;margin-top:8px"><table><thead><tr>${Object.keys(rows[0]).slice(0, 5).map(k => `<th>${esc(k)}</th>`).join('')}</tr></thead><tbody>${rows.map(r => `<tr>${Object.keys(rows[0]).slice(0, 5).map(k => `<td class="mono" style="font-size:12px">${esc(typeof r[k] === 'object' ? JSON.stringify(r[k]) : r[k])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>` : ''}`;
  }
  return h + `<p class="note" style="margin-top:8px">${fileLink('results/opposites_control_probes.json')} · ${fileLink('results/negation_control_probes.json')}</p>`;
}

/* ================================================================== CORPUS */
const CORP = { split: 'train', q: '', sort: 'count', vq: '' };
RENDER.corpus = function () {
  const sec = $('#sec-corpus'), r = run(), mf = r.manifest, vr = r.vocab_report || {};
  sec.innerHTML = `
  <div class="sec-head"><div class="intro"><span class="eyebrow">Study material · ${esc(r.label)}</span><h1>Corpus &amp; vocabulary</h1>
    <p>What this model was allowed to learn from. Duplicate passages are removed before a 90/10 split; validation passages never produce weight updates. Passages from one source file can land on both sides, so held-out loss does not measure generalization to new files.</p></div></div>
  <div class="stats">
    <div class="stat"><span class="v">${num(mf.base_passages)}</span><span class="k">generated classroom passages</span></div>
    <div class="stat"><span class="v">${num(mf.external_passages)}</span><span class="k">passages from corpus/ files</span></div>
    <div class="stat"><span class="v">${num(mf.duplicates_removed)}</span><span class="k">duplicates removed</span></div>
    <div class="stat"><span class="v">${num(mf.unique_passages)}</span><span class="k">unique passages (${num(mf.new_unique_passages)} new)</span></div>
    <div class="stat"><span class="v">${num(r.split.train.length)}<small> / ${num(r.split.validation.length)}</small></span><span class="k">train / validation</span></div>
    <div class="stat"><span class="v">${r.config.vocabulary_size}</span><span class="k">vocabulary (${num(vr.training_types)} types seen, ${num((vr.omitted_types || []).length)} cut to UNK)</span></div>
    <div class="stat"><span class="v">${pct(vr.training_unknown_rate, 2)}<small> / ${pct(vr.validation_unknown_rate, 2)}</small></span><span class="k">unknown rate, train / held-out</span></div>
    <div class="stat"><span class="v">${mf.max_passage_tokens}</span><span class="k">max tokens per passage</span></div>
  </div>
  <div class="panel" id="p-manifest">
    <div class="panel-head"><h2>Imported files</h2><p>${fileLink(`llm_runs/${r.id}/corpus_manifest.json`, 'corpus_manifest.json')} · ${fileLink(`llm_runs/${r.id}/vocabulary_report.json`, 'vocabulary_report.json')} · ${fileLink(`llm_runs/${r.id}/corpus.txt`, 'corpus.txt')}</p></div>
    ${mf.files.length ? `<div class="scroll-x"><table><thead><tr><th>File</th><th class="n">Bytes</th><th class="n">Passages</th><th class="n">Unique</th><th>Warnings</th><th>Preview</th></tr></thead><tbody>
      ${mf.files.map(f => `<tr><td class="mono">${esc(f.file)}<br><span class="faint" style="font-size:11px">sha ${esc(f.sha256.slice(0, 12))}</span></td><td class="n">${num(f.bytes)}</td><td class="n">${num(f.passages)}</td><td class="n">${num(f.unique_passages)}</td><td>${f.warnings.length ? f.warnings.map(w => `<span class="pill you">${esc(w)}</span>`).join(' ') : '<span class="faint">none</span>'}</td><td class="mono" style="font-size:11.5px;max-width:420px;white-space:pre-wrap">${esc(f.preview.slice(0, 220))}…</td></tr>`).join('')}
    </tbody></table></div>` : '<p class="note">Classroom sentences only: no files from <code>corpus/</code> were used in this run.</p>'}
    ${mf.ignored?.length ? `<p class="note" style="margin-top:8px">Ignored: ${mf.ignored.map(i => `<code>${esc(typeof i === 'string' ? i : JSON.stringify(i))}</code>`).join(', ')}</p>` : ''}
  </div>
  <div class="grid g2">
    <div class="panel" id="p-vocab">
      <div class="panel-head"><h2>Vocabulary (${r.vocab.length})</h2>
        <div class="row"><input id="voc-q" type="search" placeholder="filter…" value="${esc(CORP.vq)}" style="width:120px" aria-label="Filter vocabulary"><div class="seg"><button type="button" data-vs="count">By count</button><button type="button" data-vs="alpha">A–Z</button><button type="button" data-vs="id">By ID</button></div></div></div>
      <div class="vocab-cloud" id="voc"></div>
      <p class="note" style="margin-top:8px">Click a word to open it in the network. Number = ID. Chip size follows training frequency.</p>
    </div>
    <div class="panel" id="p-passages">
      <div class="panel-head"><h2>Passages</h2>
        <div class="row"><input id="pas-q" type="search" placeholder="search passages…" value="${esc(CORP.q)}" style="width:160px" aria-label="Search passages">
          <select id="pas-split" aria-label="Which split"><option value="train">Training (${num(r.split.train.length)})</option><option value="validation">Validation (${num(r.split.validation.length)})</option><option value="evaluation_train">Loss panel, train (${r.split.evaluation_train?.length ?? 0})</option><option value="evaluation_validation">Loss panel, validation (${r.split.evaluation_validation?.length ?? 0})</option></select></div></div>
      <div class="passages" id="pas"></div>
      <p class="note" id="pas-count" style="margin-top:6px"></p>
    </div>
  </div>
  ${vr.omitted_types?.length ? `<div class="panel"><div class="panel-head"><h2>Words cut to &lt;UNK&gt; by the 509-type limit</h2></div><div class="toks">${vr.omitted_types.slice(0, 300).map(w => `<span class="tok unk">${esc(w)}</span>`).join('')}</div></div>` : ''}`;
  const counts = r.token_counts || r.vocab.map(() => 1), maxLog = Math.log1p(Math.max(...counts));
  const drawVocab = () => {
    const q = CORP.vq.trim().toLowerCase();
    let ids = r.vocab.map((w, i) => i).filter(i => !q || r.vocab[i].includes(q));
    if (CORP.sort === 'count') ids.sort((a, b) => counts[b] - counts[a]); else if (CORP.sort === 'alpha') ids.sort((a, b) => r.vocab[a].localeCompare(r.vocab[b]));
    $('#voc').innerHTML = ids.map(i => `<span class="tok btn${/^<.+>$/.test(r.vocab[i]) ? ' special' : ''}" data-w="${esc(r.vocab[i])}" title="${num(counts[i])} occurrences" style="font-size:${(11 + 5 * Math.log1p(counts[i]) / (maxLog || 1)).toFixed(1)}px">${esc(r.vocab[i])}<sub>${i}</sub></span>`).join('');
    $$('#voc [data-w]').forEach(el => el.addEventListener('click', () => { NET.selected = el.dataset.w; go('network', null, () => selectWord(el.dataset.w, true)); }));
    $$('[data-vs]', sec).forEach(b => b.setAttribute('aria-pressed', String(b.dataset.vs === CORP.sort)));
  };
  const drawPassages = () => {
    const list = r.split[CORP.split] || [], q = CORP.q.trim().toLowerCase();
    const hits = q ? list.filter(p => p.toLowerCase().includes(q)) : list;
    const re = q ? new RegExp(q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi') : null;
    $('#pas').innerHTML = hits.slice(0, 200).map(p => `<div>${re ? esc(p).replace(new RegExp(esc(q).replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi'), s => `<mark>${s}</mark>`) : esc(p)}</div>`).join('') || '<div class="faint">No passages match.</div>';
    $('#pas-count').textContent = `${num(hits.length)} of ${num(list.length)} passages${hits.length > 200 ? ' · showing first 200' : ''}`;
  };
  $('#voc-q').addEventListener('input', e => { CORP.vq = e.target.value; drawVocab(); });
  $$('[data-vs]', sec).forEach(b => b.addEventListener('click', () => { CORP.sort = b.dataset.vs; drawVocab(); }));
  $('#pas-q').addEventListener('input', e => { CORP.q = e.target.value; drawPassages(); });
  $('#pas-split').value = CORP.split;
  $('#pas-split').addEventListener('change', e => { CORP.split = e.target.value; drawPassages(); });
  drawVocab(); drawPassages();
};

/* ================================================================== CHECKLIST */
function readmeHas(re) { return DATA.readme.some(s => re.test(s.title) || re.test(s.text)); }
function requirementStatus() {
  const C = CHECKS, nb = C.notebooks;
  const nbOK = n => n && n.errors === 0 && n.ran === n.code_cells && n.with_outputs > 0;
  const you = C.readme_placeholders > 0;
  const runsHave = f => RUNS.length >= 2 && RUNS.every(f);
  const R = (key, t, ok, go, files = [], note = '') => ({ key, t, status: ok === true ? 'done' : ok === 'you' ? 'you' : 'todo', go, files, note });
  const starter = RUNS.find(r => r.kind === 'starter'), expanded = [...RUNS].reverse().find(r => r.kind !== 'starter');
  return [
    { cat: 'Deliverable quality', pts: 4, items: [
      R('nb-starter', 'Executed starter-corpus notebook, outputs visible, no errors', nbOK(nb['custom_llm_starter.executed.ipynb']), ['training', 'p-loss'], ['custom_llm_starter.executed.ipynb'], nb['custom_llm_starter.executed.ipynb'] ? `${nb['custom_llm_starter.executed.ipynb'].ran}/${nb['custom_llm_starter.executed.ipynb'].code_cells} code cells executed, ${nb['custom_llm_starter.executed.ipynb'].with_outputs} with visible output, ${nb['custom_llm_starter.executed.ipynb'].errors} errors` : 'not found'),
      R('nb-expanded', 'Executed corpus-extension notebook, outputs visible, no errors', nbOK(nb['custom_llm_expanded.executed.ipynb']), ['training', 'p-loss'], ['custom_llm_expanded.executed.ipynb'], nb['custom_llm_expanded.executed.ipynb'] ? `${nb['custom_llm_expanded.executed.ipynb'].ran}/${nb['custom_llm_expanded.executed.ipynb'].code_cells} code cells executed, ${nb['custom_llm_expanded.executed.ipynb'].with_outputs} with visible output, ${nb['custom_llm_expanded.executed.ipynb'].errors} errors` : 'not found'),
      R('source', 'Readable source code: notebook, model, eval runner, chat interface', true, ['checklist'], ['custom_llm.ipynb', 'nanogpt_model.py', 'run_evals.py', 'chat.py']),
      R('sources', 'Corpus sources, permissions and choices explained', readmeHas(/Sources and permissions/i), ['corpus', 'p-manifest'], ['README.md', 'corpus/README.md']),
      R('ext-cats', 'At least two extension categories, with new teaching material', C.extension_files.length >= 2 && !!expanded, ['corpus', 'p-manifest'], C.extension_files.map(f => 'corpus/' + f), `files: ${C.extension_files.join(', ') || 'none'}`),
      R('prediction', 'Prediction written before training (README §2 and notebook cell)', you ? 'you' : true, ['checklist'], ['README.md'], you ? 'README still has ✍️ placeholders for you to write' : ''),
      R('explain', 'Learning process explained with your own token, embedding, gradient and loss evidence (README §8)', you ? 'you' : true, ['training', 'p-update'], ['README.md'], 'The evidence is in place. The explanation must be in your own words.'),
    ] },
    { cat: 'Testing & evaluation', pts: 3, items: [
      R('four-sets', 'Four complete result sets: starter and expanded, untrained and trained, 48 cases each', C.complete_eval_sets >= 4, ['evals', 'p-four-row'], RUNS.flatMap(r => Object.values(r.evals).map(e => e.files.json)), `${C.complete_eval_sets} complete sets of 48`),
      R('unchanged', 'Eval suite unchanged: same cases, choices, answer key and scoring', C.suite_sha256 === C.suite_sha256_expected, ['evals', 'p-leak'], ['evals/language_evals.json']),
      R('separation', 'Eval text kept out of training and vocabulary (separation files + live scan)', C.leakage.hits.length === 0 && runsHave(r => r.eval_separation), ['evals', 'p-leak'], RUNS.map(r => `llm_runs/${r.id}/eval_separation.json`), `${C.leakage.hits.length} prompt matches across ${C.leakage.sources_checked.length} training inputs`),
      R('compare', 'Compare all-case success, scorable accuracy, coverage and group/category scores', C.readme_has_four_rows, ['evals', 'p-cats'], ['README.md']),
      R('continuations', 'Free continuations saved and inspected for every case', runsHave(r => Object.values(r.evals).every(e => e.results.every(x => 'generated_text' in x))), ['evals', 'p-cases']),
      R('failures', 'Failures explained: vocabulary coverage, learned pattern, or both', readmeHas(/control probes|Coverage or learned pattern/i), ['evals', 'p-probes'], ['results/opposites_control_probes.json', 'results/negation_control_probes.json']),
    ] },
    { cat: 'Working result', pts: 3, items: [
      R('model', 'Trained nanoGPT saved for each experiment (model.pt + untrained weights)', runsHave(r => r.weights.trained && r.weights.untrained), ['lab'], RUNS.map(r => `llm_runs/${r.id}/model.pt`)),
      R('rerun', 'Evals rerunnable on saved weights with run_evals.py', !!DATA.readme.find(s => /Rerunning the evals/i.test(s.title)), ['evals', 'p-rerun'], ['run_evals.py', 'evals/README.md']),
      R('chat-code', 'Chat interface code with launch instructions', readmeHas(/chat\.py/), ['chat', 'p-chat'], ['chat.py', 'README.md']),
      R('chat-turns', 'At least 3 real chat interactions saved, including a failure', C.chat_turns >= 3, ['chat', 'p-chat-records'], C.chat_files, `${C.chat_turns} turns in the longest transcript`),
      R('chat-shot', 'Screenshot or recording of the chat', C.screenshots.length > 0, ['chat', 'p-chat-records'], C.screenshots, C.screenshots.length ? 'A rendered capture exists. A live screenshot you take yourself is stronger.' : ''),
      R('identity', 'Model/run identity shown with the chat evidence', !!DATA.chat_record?.model_sha256, ['chat', 'p-chat-records']),
    ] },
    { cat: 'README evidence', pts: null, items: [
      R('samples', 'Untrained, halfway and final samples linked, with one visible change explained', runsHave(r => Object.keys(r.samples).length >= 3) && readmeHas(/Samples/i), ['training', 'p-samples']),
      R('curves', 'training_curves.svg embedded, full loss table, panel sizes stated', runsHave(r => r.history.length > 0) && readmeHas(/Loss curves/i), ['training', 'p-loss']),
      R('token', 'Word → ID → 64-number vector, before and after', readmeHas(/Token → ID → vector/i), ['training', 'p-inspect-token'], RUNS.map(r => `llm_runs/${r.id}/inspection.json`)),
      R('gradient', 'First parameter value, gradient and update explained', readmeHas(/gradient/i), ['training', 'p-update']),
      R('probs', 'Next-token probability comparison', readmeHas(/Next-token probabilities/i), ['training', 'p-probs']),
      R('temperature', 'Temperature comparison, and what changes only at inference', readmeHas(/Temperature/i), ['training', 'p-temp']),
      R('limitation', 'One observed limitation and one proposed next experiment (README §9)', you ? 'you' : true, ['checklist'], ['README.md'], 'Confirm the §9 prediction is yours.'),
    ] },
    { cat: 'Submission', pts: null, items: [
      R('push', 'Push to a public GitHub repository', C.git, ['checklist'], [], C.git ? '' : 'No git repository yet'),
      R('signed-out', 'Open the repo signed out: notebooks, plot and evidence links all load', false, ['checklist']),
      R('submit', 'Submit the repository URL on bCourses', false, ['checklist']),
    ] },
  ];
}
RENDER.checklist = function () {
  const sec = $('#sec-checklist'), groups = requirementStatus();
  const all = groups.flatMap(g => g.items), n = s => all.filter(i => i.status === s).length;
  sec.innerHTML = `
  <div class="sec-head"><div class="intro"><span class="eyebrow">Graded out of 10</span><h1>Assignment checklist</h1>
    <p>Each requirement from the brief, checked against the files in this project when the dashboard was built. “Needs your writing” marks sections the assignment asks you to explain in your own words.</p></div>
    <div class="row"><span class="pill done">${n('done')} done</span><span class="pill you">${n('you')} need your writing</span><span class="pill todo">${n('todo')} not yet</span></div></div>
  ${groups.map(g => `<div class="panel"><div class="panel-head"><h2>${esc(g.cat)}${g.pts ? ` <span class="faint" style="font-weight:500">· ${g.pts} points</span>` : ''}</h2></div>
    <div class="req-group">${g.items.map(i => `<div class="req" id="req-${i.key}">
      <div><span class="pill ${i.status}">${i.status === 'done' ? 'Done' : i.status === 'you' ? 'Needs your writing' : 'Not yet'}</span></div>
      <div class="t">${esc(i.t)}</div>
      <div class="ev">${i.note ? `<span>${esc(i.note)}</span>` : ''}${i.files.slice(0, 6).map(f => fileLink(f)).join('')}${i.go[0] !== 'checklist' ? `<a href="#${i.go[0]}" data-jump="${i.go.join('|')}">See it in the dashboard →</a>` : ''}</div>
    </div>`).join('')}</div></div>`).join('')}
  <div class="panel" id="p-brief">
    <div class="panel-head"><h2>The full brief</h2><p>From <code>ASSIGNMENT.md</code>. Every section is also in the search (⌘K).</p></div>
    ${DATA.assignment.map((s, i) => `<details id="brief-${i}" style="border-top:1px solid var(--line);padding:8px 0"><summary><b>${esc(s.title)}</b></summary><div class="note" style="white-space:pre-wrap;margin-top:6px;max-width:80ch">${esc(s.text)}</div></details>`).join('')}
  </div>`;
  $$('[data-jump]', sec).forEach(a => a.addEventListener('click', e => { e.preventDefault(); const [s, id] = a.dataset.jump.split('|'); go(s, id); }));
};

/* ================================================================== CONCEPTS */
function concepts() {
  const r = run(), c = r.config, ins = r.inspection, last = r.history[r.history.length - 1], first = r.history[0];
  return [
    ['corpus', 'Corpus', 'The study material: every passage the model learns from. Here, generated classroom sentences plus files in corpus/.', `${num(r.manifest.unique_passages)} unique passages in this run.`, ['corpus', 'p-manifest']],
    ['passage', 'Passage (document)', 'One short training example of at most 47 word/punctuation tokens. Long files are split into passages at sentence or line boundaries.', `${num(c.train_documents)} training and ${num(c.validation_documents)} validation passages.`, ['corpus', 'p-passages']],
    ['token', 'Token', 'The unit the model reads: here a whole word or a punctuation mark, lowercased. Not characters, not sub-word pieces.', `“${ins.token}” is one token.`, ['training', 'p-inspect-token']],
    ['vocabulary', 'Vocabulary and token IDs', 'The list of token types the model knows. Each gets an integer ID, which is just its row number in the embedding table. It is built only from training text and capped at 509 types.', `${c.vocabulary_size} types; “${ins.token}” is ID ${ins.token_id}.`, ['corpus', 'p-vocab']],
    ['unk', 'Unknown token (<UNK>)', 'Any word outside the vocabulary becomes <UNK>. The model cannot learn anything specific about it, which is why eval cases with unknown words are unscorable.', `Held-out unknown rate ${pct(c.validation_unknown_rate, 2)}.`, ['corpus', 'p-vocab']],
    ['embedding', 'Embedding (vector)', 'A learned list of 64 numbers for each token. Training nudges these numbers so that tokens used in similar contexts get similar vectors.', `The “${ins.token}” vector is shown before and after on the Training page.`, ['network']],
    ['cosine', 'Cosine similarity', 'How closely two vectors point the same way, from −1 to 1. The word network links each word to its highest-cosine neighbours.', 'Uses all 64 dimensions, unlike the 2D PCA map.', ['network']],
    ['position', 'Positional embedding', 'A second learned vector for each position 0–47, added to the token vector so the model knows word order.', `${c.block_size} positions × ${c.n_embd} numbers.`, ['lab', 'p-flow']],
    ['attention', 'Self-attention', 'At each position, the model scores every earlier position, turns the scores into weights that sum to 1, and takes a weighted mix of their vectors. That is how earlier context reaches the prediction. It cannot look ahead.', `${c.n_layer} blocks × ${c.n_head} heads.`, ['lab', 'p-heat']],
    ['head', 'Attention head', 'One of several attention patterns computed in parallel inside a block, each on a 16-number slice. Different heads can track different relationships.', 'Compare heads side by side in the lab.', ['lab', 'p-heat']],
    ['mlp', 'MLP (feed-forward layer)', 'After attention, each position passes through a small two-layer network (64 → 256 → 64 with GELU). The 256 hidden units are the neurons in the neuron map.', `${c.n_layer} MLPs, 256 hidden units each.`, ['lab', 'p-neurons']],
    ['weights', 'Weights / parameters', 'Every learned number in the network: embeddings, attention and MLP matrices, LayerNorm gains. Training changes only these.', `${num(c.parameters)} parameters.`, ['lab']],
    ['logits', 'Logits and softmax', 'The final vector is compared with every token embedding (weight tying), giving one score (logit) per word. Softmax turns scores into probabilities that sum to 1.', `${c.vocabulary_size} probabilities per prediction.`, ['lab', 'p-next']],
    ['loss', 'Loss (cross-entropy)', 'How surprised the model was by the real next word: −log(probability it gave that word), averaged. Lower is better. A uniform guess scores ln(V).', `Validation panel ${fx(first?.validation_loss)} → ${fx(last?.validation_loss)}; ln(${c.vocabulary_size}) = ${Math.log(c.vocabulary_size).toFixed(3)}.`, ['training', 'p-loss']],
    ['gradient', 'Gradient (backpropagation)', 'For each weight: how much the loss would change if that weight went up slightly. Backpropagation computes all of them in one backward pass.', ins.first_update ? `First recorded gradient: ${sci(ins.first_update.gradient)}.` : '', ['training', 'p-update']],
    ['lr', 'Learning rate, warmup, cosine decay', 'The size of each optimizer step. It ramps up over the first 100 steps (warmup), then decays along a cosine curve to 10% of peak. Too large and training diverges; too small and it barely moves.', `Peak ${c.learning_rate}; step-1 rate ${ins.first_update ? sci(ins.first_update.learning_rate) : '—'}.`, ['training', 'p-update']],
    ['adamw', 'AdamW optimizer', 'Adjusts each weight using running averages of its gradients (momentum) and their squares (scale), plus a small weight decay. Its first step moves each weight by about the learning rate.', 'betas (0.9, 0.95), weight decay 0.01.', ['training', 'p-update']],
    ['step', 'Training step', 'One weight update from a batch of 32 passages. A step is not a pass through the whole corpus.', `${num(c.training_steps)} steps in ${fx(r.summary?.elapsed_seconds, 1)} s.`, ['training']],
    ['validation', 'Training vs validation loss', 'Training loss is measured on passages the model learns from; validation loss on held-out passages it never updates on. A widening gap suggests memorization.', `Final gap ${fx(last ? last.validation_loss - last.training_loss : null, 4)}.`, ['training', 'p-loss']],
    ['temperature', 'Temperature', 'Divides the logits before softmax at generation time. Below 1 sharpens toward the top word, above 1 flattens toward randomness. It never changes the weights.', 'Try the slider in the lab or chat.', ['training', 'p-temp']],
    ['context', 'Context window', 'The most tokens the model can see at once: 48. Longer prompts keep only the most recent 48.', `${c.block_size} tokens.`, ['chat']],
    ['eval', 'Eval, scorable, coverage', 'A fixed test: a prompt, four one-word choices, an answer key. Scorable means every word is in the vocabulary; coverage is the share of cases that are scorable.', `${evalStats(r, 'final')?.correct ?? '—'}/48 correct, ${evalStats(r, 'final')?.scorable ?? '—'} scorable in this run.`, ['evals', 'p-four-row']],
    ['continuation', 'Free continuation', 'The unrestricted text the model generates after an eval prompt. It can disagree with its own multiple-choice pick, and it is saved for inspection, not scored.', '', ['evals', 'p-cases']],
    ['leakage', 'Leakage', 'Eval prompts, answers or outputs ending up in training data. It turns memorization into a fake score and carries a grade penalty.', `${CHECKS.leakage.hits.length} matches in the live scan.`, ['evals', 'p-leak']],
  ];
}
RENDER.glossary = function () {
  const sec = $('#sec-glossary');
  sec.innerHTML = `
  <div class="sec-head"><div class="intro"><span class="eyebrow">Reference</span><h1>Concepts, with this run's numbers</h1>
    <p>Short definitions tied to the live evidence. The README asks for these ideas explained in your own words; use this page to check your understanding, not as text to copy.</p></div></div>
  <div class="grid" style="grid-template-columns:repeat(auto-fill,minmax(300px,1fr))">
    ${concepts().map(([k, t, d, live, g]) => `<div class="panel stack" id="concept-${k}" style="gap:8px"><h3>${esc(t)}</h3><p class="note" style="font-size:13px;color:var(--ink)">${esc(d)}</p>${live ? `<p class="note mono" style="font-size:12px">${esc(live)}</p>` : ''}<a href="#${g[0]}" data-jump="${g.join('|')}" style="font-size:12.5px;margin-top:auto">See it →</a></div>`).join('')}
  </div>`;
  $$('[data-jump]', sec).forEach(a => a.addEventListener('click', e => { e.preventDefault(); const [s, id] = a.dataset.jump.split('|'); go(s, id); }));
};

/* ================================================================== SEARCH */
const SEARCH = { scope: 'all', sel: 0, flat: [] };
const KINDS = ['Requirement', 'Concept', 'Page', 'Eval case', 'Word', 'Assignment', 'README', 'Sample', 'File', 'Passage'];
const PAGES = [
  ['overview', 'Overview', 'experiments, four runs, progress, adding data'], ['network', 'Word network', 'embeddings, neighbours, cosine similarity, clusters, drag'],
  ['lab', 'Transformer lab', 'attention maps, heads, information flow, next-word probabilities, neurons, generate'], ['chat', 'Chat', 'chat interface, transcript, terminal, chat.py'],
  ['training', 'Training evidence', 'loss curves, samples, gradient, weight update, temperature, tokenization'], ['evals', 'Evals', '48 cases, four-row comparison, categories, leakage, rerun'],
  ['corpus', 'Corpus & vocabulary', 'manifest, passages, vocabulary, unknown rate, split'], ['checklist', 'Assignment checklist', 'requirements, grading, submission, brief'], ['glossary', 'Concepts', 'definitions, glossary'],
];
function buildIndex() {
  const r = run(), idx = [];
  requirementStatus().forEach(g => g.items.forEach(i => idx.push({ kind: 'Requirement', title: i.t, text: `${g.cat} · ${i.status === 'done' ? 'done' : i.status === 'you' ? 'needs your writing' : 'not yet'} ${i.note} ${i.files.join(' ')}`, act: () => go('checklist', 'req-' + i.key), badge: i.status })));
  concepts().forEach(([k, t, d, live]) => idx.push({ kind: 'Concept', title: t, text: d + ' ' + live, act: () => go('glossary', 'concept-' + k) }));
  PAGES.forEach(([s, t, d]) => idx.push({ kind: 'Page', title: t, text: d, act: () => go(s) }));
  CASES.forEach(c => idx.push({ kind: 'Eval case', title: `${c.id} · ${c.prompt} ___`, text: `${c.category} ${c.group} answer: ${c.answer} choices: ${c.choices.join(', ')} ${c.reason || ''}`, act: () => { EV.open = c.id; EV.q = ''; EV.group = 'all'; EV.cat = 'all'; EV.outcome = 'all'; S.rendered.evals = null; go('evals', 'case-' + c.id); } }));
  r.vocab.forEach((w, i) => idx.push({ kind: 'Word', title: w, text: `token id ${i}`, act: () => { NET.selected = w; go('network', null, () => selectWord(w, true)); } }));
  DATA.assignment.forEach((s, i) => idx.push({ kind: 'Assignment', title: s.title, text: s.text, act: () => go('checklist', 'brief-' + i, () => { const d = document.getElementById('brief-' + i); if (d) { d.open = true; d.scrollIntoView({ block: 'start', behavior: 'smooth' }); } }) }));
  DATA.readme.forEach(s => idx.push({ kind: 'README', title: s.title, text: s.text, act: () => { const href = fileHref('README.md'); if (href) window.open(href, '_blank', 'noopener'); else go('checklist'); } }));
  RUNS.forEach(rr => Object.entries(rr.samples).forEach(([st, lines]) => lines.forEach(l => idx.push({ kind: 'Sample', title: l || '(empty)', text: `${rr.label} step ${st}`, act: () => { S.run = RUNS.indexOf(rr); $('#run-select').value = S.run; S.rendered = {}; go('training', 'p-samples'); } }))));
  RUNS.forEach(rr => rr.files.forEach(f => idx.push({ kind: 'File', title: f.path, text: `${rr.label} ${num(f.bytes)} bytes`, act: () => { const h = fileHref(f.path); if (h) window.open(h, '_blank', 'noopener'); } })));
  ['README.md', 'ASSIGNMENT.md', 'chat.py', 'run_evals.py', 'custom_llm.ipynb', 'nanogpt_model.py', 'evals/language_evals.json', 'evals/README.md', 'build_dashboard.py', ...CHECKS.chat_files, ...CHECKS.screenshots, ...CHECKS.extension_files.map(f => 'corpus/' + f)].forEach(p => idx.push({ kind: 'File', title: p, text: 'project file', act: () => { const h = fileHref(p); if (h) window.open(h, '_blank', 'noopener'); } }));
  ['train', 'validation'].forEach(sp => r.split[sp].forEach(p => idx.push({ kind: 'Passage', title: p, text: `${sp} passage · ${r.label}`, act: () => { CORP.q = p.split(' ').slice(0, 6).join(' '); CORP.split = sp; S.rendered.corpus = null; go('corpus', 'p-passages'); } })));
  return idx;
}
let INDEX = null, INDEX_RUN = null;
function hl(text, terms) { let s = esc(text); terms.forEach(t => { if (t.length < 2) return; s = s.replace(new RegExp(esc(t).replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi'), m => `<mark>${m}</mark>`); }); return s; }
function snippet(text, terms) {
  const low = text.toLowerCase(); let at = -1; for (const t of terms) { at = low.indexOf(t); if (at >= 0) break; }
  const s = Math.max(0, at - 60); return (s > 0 ? '…' : '') + text.slice(s, s + 200).replace(/\s+/g, ' ') + (text.length > s + 200 ? '…' : '');
}
function runSearch(q) {
  if (!INDEX || INDEX_RUN !== S.run) { INDEX = buildIndex(); INDEX_RUN = S.run; }
  const terms = q.toLowerCase().split(/\s+/).filter(Boolean);
  const res = $('#search-results');
  const kinds = SEARCH.scope === 'all' ? KINDS : [SEARCH.scope];
  let groups = [];
  if (!terms.length) {
    groups = [['Page', INDEX.filter(i => i.kind === 'Page')], ['Requirement', INDEX.filter(i => i.kind === 'Requirement' && i.badge !== 'done')]];
  } else {
    const scored = INDEX.filter(i => kinds.includes(i.kind) && !(i.kind === 'Passage' && q.length < 3)).map(i => {
      const T = i.title.toLowerCase(), X = i.text.toLowerCase(); let s = 0;
      for (const t of terms) { if (T === t) s += 40; else if (T.startsWith(t)) s += 16; else if (T.includes(t)) s += 10; else if (X.includes(t)) s += 3; else return null; }
      return { i, s: s - (i.kind === 'Word' && T !== terms.join(' ') ? 8 : 0) };
    }).filter(Boolean);
    groups = KINDS.filter(k => kinds.includes(k)).map(k => [k, scored.filter(x => x.i.kind === k).sort((a, b) => b.s - a.s).slice(0, k === 'Requirement' ? 8 : k === 'Passage' ? 6 : SEARCH.scope === 'all' ? 6 : 40).map(x => x.i)]).filter(([, l]) => l.length);
    groups.sort((a, b) => { const best = l => Math.max(...l.map(i => { const T = i.title.toLowerCase(); return T === terms.join(' ') ? 3 : T.includes(terms[0]) ? 2 : 1; })); return best(b[1]) - best(a[1]) || KINDS.indexOf(a[0]) - KINDS.indexOf(b[0]); });
  }
  SEARCH.flat = groups.flatMap(([, l]) => l); SEARCH.sel = 0;
  let n = 0;
  res.innerHTML = groups.length ? groups.map(([k, l]) => `<div class="sr-group eyebrow">${esc(k)}${k === 'Passage' ? ` · ${esc(run().label)}` : ''}</div>${l.map(i => `<div class="sr" role="option" data-i="${n++}" aria-selected="false"><span class="st">${hl(i.title.length > 140 ? i.title.slice(0, 140) + '…' : i.title, terms)}${i.badge ? ` <span class="pill ${i.badge}" style="margin-left:6px">${i.badge === 'done' ? 'done' : i.badge === 'you' ? 'your words' : 'not yet'}</span>` : ''}</span>${i.text ? `<span class="ss">${hl(snippet(i.text, terms), terms)}</span>` : ''}</div>`).join('')}`).join('')
    : `<div class="sr-empty">Nothing matches “${esc(q)}”. Try a single word such as <b>loss</b>, <b>negation</b> or <b>README</b>, or switch the scope to All.</div>`;
  $$('.sr', res).forEach(el => { el.addEventListener('click', () => pick(+el.dataset.i)); el.addEventListener('mousemove', () => mark(+el.dataset.i)); });
  mark(0);
}
function mark(i) { SEARCH.sel = i; $$('.sr').forEach(el => el.setAttribute('aria-selected', String(+el.dataset.i === i))); $(`.sr[data-i="${i}"]`)?.scrollIntoView({ block: 'nearest' }); }
function pick(i) { const it = SEARCH.flat[i]; if (!it) return; closeSearch(); it.act(); }
function openSearch(q = '') {
  $('#search').hidden = false; const inp = $('#search-input'); inp.value = q; inp.focus(); runSearch(q);
}
function closeSearch() { $('#search').hidden = true; }
function setupSearch() {
  $('#search-scopes').innerHTML = ['all', ...KINDS].map(k => `<button class="chip" data-scope="${k}" aria-pressed="${k === SEARCH.scope}">${k === 'all' ? 'All' : k}</button>`).join('');
  $$('[data-scope]').forEach(b => b.addEventListener('click', () => { SEARCH.scope = b.dataset.scope; $$('[data-scope]').forEach(x => x.setAttribute('aria-pressed', String(x === b))); runSearch($('#search-input').value); $('#search-input').focus(); }));
  $('#search-open').addEventListener('click', () => openSearch());
  $('#search').addEventListener('click', e => { if (e.target.id === 'search') closeSearch(); });
  $('#search-input').addEventListener('input', e => runSearch(e.target.value));
  $('#search-input').addEventListener('keydown', e => {
    if (e.key === 'ArrowDown') { e.preventDefault(); mark(Math.min(SEARCH.flat.length - 1, SEARCH.sel + 1)); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); mark(Math.max(0, SEARCH.sel - 1)); }
    else if (e.key === 'Enter') { e.preventDefault(); pick(SEARCH.sel); }
  });
  document.addEventListener('keydown', e => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); $('#search').hidden ? openSearch() : closeSearch(); }
    else if (e.key === '/' && $('#search').hidden && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName)) { e.preventDefault(); openSearch(); }
    else if (e.key === 'Escape' && !$('#search').hidden) closeSearch();
  });
}

/* ================================================================== boot */
setupShell();
setupSearch();
go(SECTIONS.includes(location.hash.slice(1)) ? location.hash.slice(1) : 'overview');
