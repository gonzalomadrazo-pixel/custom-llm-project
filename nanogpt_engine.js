/* nanoGPT forward pass in plain JavaScript.
   Mirrors nanogpt_model.py: token+positional embeddings, N blocks of
   (LayerNorm -> causal multi-head self-attention -> residual,
    LayerNorm -> MLP(4x, GELU) -> residual), final LayerNorm,
   output head tied to the token embedding table. */
function makeModel(BUNDLE) {
  const cfg = BUNDLE.config, W = BUNDLE.weights;
  const NE = cfg.n_embd, NH = cfg.n_head, NL = cfg.n_layer, HS = NE / NH, BS = cfg.block_size;
  const V = cfg.vocab, vocabSize = V.length;
  const stoi = new Map(V.map((t, i) => [t, i]));

  const erf = x => { // Abramowitz & Stegun 7.1.26, |error| < 1.5e-7
    const s = x < 0 ? -1 : 1; x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  };
  const gelu = x => 0.5 * x * (1 + erf(x / Math.SQRT2)); // PyTorch nn.GELU default (exact)

  function layerNorm(v, w, b) { // v: Float64Array(NE)
    let m = 0; for (let i = 0; i < NE; i++) m += v[i]; m /= NE;
    let s = 0; for (let i = 0; i < NE; i++) { const d = v[i] - m; s += d * d; } s /= NE;
    const inv = 1 / Math.sqrt(s + 1e-5), out = new Float64Array(NE);
    for (let i = 0; i < NE; i++) out[i] = (v[i] - m) * inv * w[i] + b[i];
    return out;
  }
  // PyTorch Linear stores weight as [out, in]; y = W x + b
  function linear(x, w, b, nIn, nOut) {
    const out = new Float64Array(nOut);
    for (let o = 0; o < nOut; o++) {
      let acc = b ? b[o] : 0, base = o * nIn;
      for (let i = 0; i < nIn; i++) acc += w[base + i] * x[i];
      out[o] = acc;
    }
    return out;
  }

  /* Full forward over the sequence; returns logits for the final position. */
  function forward(ids) {
    const T = Math.min(ids.length, BS);
    const seq = ids.slice(ids.length - T); // keep the most recent BS tokens
    const wte = W["transformer.wte.weight"], wpe = W["transformer.wpe.weight"];
    let x = [];
    for (let t = 0; t < T; t++) {
      const v = new Float64Array(NE), tb = seq[t] * NE, pb = t * NE;
      for (let i = 0; i < NE; i++) v[i] = wte[tb + i] + wpe[pb + i];
      x.push(v);
    }
    for (let l = 0; l < NL; l++) {
      const P = `transformer.h.${l}.`;
      const ln1w = W[P + "ln_1.weight"], ln1b = W[P + "ln_1.bias"];
      const caW = W[P + "attn.c_attn.weight"], caB = W[P + "attn.c_attn.bias"];
      const cpW = W[P + "attn.c_proj.weight"], cpB = W[P + "attn.c_proj.bias"];
      const ln2w = W[P + "ln_2.weight"], ln2b = W[P + "ln_2.bias"];
      const fcW = W[P + "mlp.c_fc.weight"], fcB = W[P + "mlp.c_fc.bias"];
      const pjW = W[P + "mlp.c_proj.weight"], pjB = W[P + "mlp.c_proj.bias"];

      // --- attention ---
      const q = [], k = [], v = [];
      for (let t = 0; t < T; t++) {
        const h = layerNorm(x[t], ln1w, ln1b);
        const qkv = linear(h, caW, caB, NE, 3 * NE);
        q.push(qkv.slice(0, NE)); k.push(qkv.slice(NE, 2 * NE)); v.push(qkv.slice(2 * NE, 3 * NE));
      }
      const attnOut = [];
      for (let t = 0; t < T; t++) {
        const y = new Float64Array(NE);
        for (let h = 0; h < NH; h++) {
          const off = h * HS, scores = new Float64Array(t + 1);
          let mx = -Infinity;
          for (let s = 0; s <= t; s++) { // causal: only positions <= t
            let d = 0;
            for (let i = 0; i < HS; i++) d += q[t][off + i] * k[s][off + i];
            d /= Math.sqrt(HS); scores[s] = d; if (d > mx) mx = d;
          }
          let sum = 0;
          for (let s = 0; s <= t; s++) { scores[s] = Math.exp(scores[s] - mx); sum += scores[s]; }
          for (let s = 0; s <= t; s++) {
            const a = scores[s] / sum;
            for (let i = 0; i < HS; i++) y[off + i] += a * v[s][off + i];
          }
        }
        attnOut.push(linear(y, cpW, cpB, NE, NE));
      }
      for (let t = 0; t < T; t++) for (let i = 0; i < NE; i++) x[t][i] += attnOut[t][i];

      // --- MLP ---
      for (let t = 0; t < T; t++) {
        const h = layerNorm(x[t], ln2w, ln2b);
        const f = linear(h, fcW, fcB, NE, 4 * NE);
        for (let i = 0; i < 4 * NE; i++) f[i] = gelu(f[i]);
        const p = linear(f, pjW, pjB, 4 * NE, NE);
        for (let i = 0; i < NE; i++) x[t][i] += p[i];
      }
    }
    const hF = layerNorm(x[T - 1], W["transformer.ln_f.weight"], W["transformer.ln_f.bias"]);
    return linear(hF, W["lm_head.weight"], null, NE, vocabSize); // head tied to wte
  }

  const tokenize = s => (s.toLowerCase().match(/[0-9a-z_]+(?:['’][0-9a-z_]+)*|[^\sa-z0-9_]/g) || []);
  const softmax = (logits, temp) => {
    const t = Math.max(temp, 1e-6), out = new Float64Array(logits.length);
    let mx = -Infinity;
    for (let i = 0; i < logits.length; i++) { out[i] = logits[i] / t; if (out[i] > mx) mx = out[i]; }
    let sum = 0;
    for (let i = 0; i < out.length; i++) { out[i] = Math.exp(out[i] - mx); sum += out[i]; }
    for (let i = 0; i < out.length; i++) out[i] /= sum;
    return out;
  };

  function encode(text) {
    const toks = tokenize(text), unknown = [];
    const ids = toks.map(t => {
      if (stoi.has(t)) return stoi.get(t);
      if (!unknown.includes(t)) unknown.push(t);
      return 0; // <UNK>
    });
    return { ids, unknown, toks };
  }

  function generate(text, { temperature = 0.8, maxTokens = 24, greedy = false, rand = Math.random } = {}) {
    const { ids, unknown, toks } = encode(text);
    let seq = [1].concat(ids); // <BOS>
    const truncated = seq.length > BS;
    const out = [];
    for (let n = 0; n < maxTokens; n++) {
      const logits = forward(seq);
      let nxt;
      if (greedy) {
        nxt = 0; for (let i = 1; i < logits.length; i++) if (logits[i] > logits[nxt]) nxt = i;
      } else {
        const p = softmax(logits, temperature);
        let r = rand(), acc = 0; nxt = p.length - 1;
        for (let i = 0; i < p.length; i++) { acc += p[i]; if (r < acc) { nxt = i; break; } }
      }
      if (nxt === 2) break;             // <EOS>
      seq.push(nxt); out.push(V[nxt]);
    }
    return { text: out.join(" "), unknown, truncated, promptTokens: toks.length };
  }

  function nextTokenTable(text, temperature = 1.0, k = 8) {
    const { ids } = encode(text);
    const p = softmax(forward([1].concat(ids)), temperature);
    return Array.from(p, (prob, i) => [V[i], prob])
      .sort((a, b) => b[1] - a[1]).slice(0, k);
  }

  return { forward, generate, encode, nextTokenTable, vocab: V, blockSize: BS, stoi };
}
if (typeof module !== "undefined") module.exports = { makeModel };
