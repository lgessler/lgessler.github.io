/* ════════════════════════════════════════════════════════════════
   HC3 Text Classification Explorer — app.js
   LING-L 245 · Naive Bayes + Perceptron on HC3 Reddit subset
   ════════════════════════════════════════════════════════════════ */

'use strict';

// ── Constants ────────────────────────────────────────────────────
const VOCAB_SIZE  = 5000;
const SEED        = 12345;
const TEST_RATIO  = 0.20;
const TOP_CONF_N  = 20;   // rows shown in "Most Confident" filters

// Threshold slider lookup tables (13 values each)
const NB_THRESH_VALUES   = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0];  // log-odds
const PERC_THRESH_VALUES = [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0,  0.5,  1.0,  1.5,  2.0,  2.5,  3.0];

// ── Global state ─────────────────────────────────────────────────
let DATASET   = null;
let TRAIN_ALL = null;
let TEST_ALL  = null;

// Per-model state
const state = {
  nb:   { model: null, mode: null, threshold: 1.0, rawResults: null, filter: 'all',
          wordScores: null, weightsDirFilter: 'all', ngramFilter: null },
  perc: { model: null, mode: null, threshold: 0.0, rawResults: null, filter: 'all',
          wordScores: null, weightsDirFilter: 'all', ngramFilter: null }
};

const charts = {};

// ── Seeded PRNG (Mulberry32) ─────────────────────────────────────
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function seededShuffle(arr, rng) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// ── Tokenisation / feature extraction ───────────────────────────
function tokenize(text) {
  // Normalise curly apostrophes to straight so contractions aren't split.
  text = text.toLowerCase().replace(/[\u2018\u2019]/g, "'");
  return text.match(/[a-z0-9]+(?:'[a-z]+)*/g) || [];
}

function extractFeatures(text, mode) {
  const tokens = tokenize(text);
  const freq = {};
  for (const tok of tokens) freq[tok] = (freq[tok] || 0) + 1;
  if (mode === 'bigram') {
    for (let i = 0; i < tokens.length - 1; i++) {
      const bg = tokens[i] + '_' + tokens[i + 1];
      freq[bg] = (freq[bg] || 0) + 1;
    }
  }
  return freq;
}

function buildVocab(examples, mode) {
  const counts = {};
  for (const ex of examples) {
    for (const [f, c] of Object.entries(extractFeatures(ex.text, mode)))
      counts[f] = (counts[f] || 0) + c;
  }
  const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  return new Set(sorted.slice(0, VOCAB_SIZE).map(([f]) => f));
}

// ── Data loading & splitting ─────────────────────────────────────
async function loadData() {
  const resp = await fetch('data/raid-reddit.json');
  if (!resp.ok) throw new Error(`HTTP ${resp.status} loading data`);
  DATASET = await resp.json();

  const rng      = mulberry32(SEED);
  const shuffled = seededShuffle(DATASET.examples, rng);

  const byLabel = [[], []];
  for (const ex of shuffled) byLabel[ex.label].push(ex);

  const train = [], test = [];
  for (const group of byLabel) {
    const nTest = Math.round(group.length * TEST_RATIO);
    test.push(...group.slice(0, nTest));
    train.push(...group.slice(nTest));
  }
  TRAIN_ALL = seededShuffle(train, rng);
  TEST_ALL  = seededShuffle(test,  rng);
}

// ── NaiveBayes ───────────────────────────────────────────────────
class NaiveBayes {
  constructor() {
    this.logPrior  = {};
    this.logLik    = {};
    this.vocab     = null;
    this.vocabList = [];
    this._mode     = 'unigram';
  }

  train(examples, vocab, mode, k = 1) {
    this._mode     = mode;
    this.vocab     = vocab;
    this.vocabList = Array.from(vocab);
    const classDocs = {0: [], 1: []};
    for (const ex of examples) classDocs[ex.label].push(ex);

    const N = examples.length;
    for (const lbl of [0, 1])
      this.logPrior[lbl] = Math.log(classDocs[lbl].length / N);

    for (const lbl of [0, 1]) {
      const termCounts = {};
      let total = 0;
      for (const ex of classDocs[lbl]) {
        for (const [f, c] of Object.entries(extractFeatures(ex.text, mode))) {
          if (vocab.has(f)) { termCounts[f] = (termCounts[f] || 0) + c; total += c; }
        }
      }
      const V     = vocab.size;
      const denom = Math.log(total + k * V);
      this.logLik[lbl] = {};
      for (const f of vocab)
        this.logLik[lbl][f] = Math.log((termCounts[f] || 0) + k) - denom;
    }
  }

  // Returns { odds } — label decision made externally by threshold
  rawScore(features) {
    const s = {};
    for (const lbl of [0, 1]) {
      s[lbl] = this.logPrior[lbl];
      for (const [f, c] of Object.entries(features))
        if (this.vocab.has(f) && this.logLik[lbl][f] !== undefined)
          s[lbl] += c * this.logLik[lbl][f];
    }
    return { odds: s[1] - s[0] };  // log-odds; avoids Math.exp underflow to 0
  }

  topWords(n) {
    const scored = this.vocabList.map(f => ({
      word:  f,
      score: (this.logLik[1][f] || 0) - (this.logLik[0][f] || 0)
    }));
    scored.sort((a, b) => b.score - a.score);
    return { top: scored.slice(0, n), bottom: scored.slice(-n).reverse() };
  }
}

// ── Perceptron ───────────────────────────────────────────────────
class Perceptron {
  constructor() {
    this.weights = {};
    this.bias    = 0;
    this._mode   = 'unigram';
  }

  _dot(features) {
    let s = this.bias;
    for (const [f, c] of Object.entries(features))
      if (this.weights[f]) s += this.weights[f] * c;
    return s;
  }

  trainEpoch(examples, lr) {
    let correct = 0;
    for (const ex of examples) {
      const feats = extractFeatures(ex.text, this._mode);
      const score = this._dot(feats);
      const pred  = score >= 0 ? 1 : 0;
      if (pred !== ex.label) {
        const delta = lr * (ex.label === 1 ? 1 : -1);
        this.bias += delta;
        for (const [f, c] of Object.entries(feats))
          this.weights[f] = (this.weights[f] || 0) + delta * c;
      } else {
        correct++;
      }
    }
    return correct / examples.length;
  }

  train(trainExamples, testExamples, epochs, lr, mode) {
    this._mode   = mode;
    this.weights = {};
    this.bias    = 0;
    const epochTrainAccs = [];
    const epochTestAccs  = [];
    const rng = mulberry32(SEED + 7);
    let exs = trainExamples.slice();
    for (let e = 0; e < epochs; e++) {
      exs = seededShuffle(exs, rng);
      epochTrainAccs.push(this.trainEpoch(exs, lr));
      let correct = 0;
      for (const ex of testExamples)
        if ((this._dot(extractFeatures(ex.text, mode)) >= 0 ? 1 : 0) === ex.label) correct++;
      epochTestAccs.push(correct / testExamples.length);
    }
    return { epochTrainAccs, epochTestAccs };
  }

  // Returns { score } — label decision made externally by threshold
  rawScore(features) {
    return { score: this._dot(features) };
  }

  topWords(n) {
    const entries = Object.entries(this.weights);
    entries.sort((a, b) => b[1] - a[1]);
    return {
      top:    entries.slice(0, n).map(([word, score]) => ({ word, score })),
      bottom: entries.slice(-n).reverse().map(([word, score]) => ({ word, score }))
    };
  }
}

// ── Raw results & threshold application ──────────────────────────
// Compute once after training; stored so threshold changes don't retrain.
function computeRawResults(examples, model, mode) {
  const isNB = model instanceof NaiveBayes;
  return examples.map(ex => {
    const feats = extractFeatures(ex.text, mode);
    const raw   = model.rawScore(feats);
    return {
      text:      ex.text,
      trueLabel: ex.label,
      rawScore:  isNB ? raw.odds : raw.score,
      isNB
    };
  });
}

function applyThreshold(rawResults, threshold) {
  return rawResults.map(r => {
    const predLabel  = r.rawScore >= threshold ? 1 : 0;
    const correct    = predLabel === r.trueLabel;
    // Confidence = how far the raw score is from the threshold
    const confidence = Math.abs(r.rawScore - threshold);
    return { ...r, predLabel, correct, confidence };
  });
}

// ── Metrics ──────────────────────────────────────────────────────
function metricsFromResults(results) {
  let TP = 0, TN = 0, FP = 0, FN = 0;
  for (const r of results) {
    const g = r.trueLabel, p = r.predLabel;
    if      (g === 1 && p === 1) TP++;
    else if (g === 0 && p === 0) TN++;
    else if (g === 0 && p === 1) FP++;
    else                         FN++;
  }
  const n = results.length;
  return {
    accuracy:    (TP + TN) / n,
    sensitivity: (TP + FN) > 0 ? TP / (TP + FN) : 0,
    specificity: (TN + FP) > 0 ? TN / (TN + FP) : 0
  };
}

function renderMetrics(containerId, trainResults, testResults) {
  const trainM = metricsFromResults(trainResults);
  const testM  = metricsFromResults(testResults);
  function pct(v) { return (v * 100).toFixed(1) + '%'; }
  function row(label, m) {
    return `<tr>
      <td class="mrow-label">${label}</td>
      <td>${pct(m.accuracy)}</td>
      <td>${pct(m.sensitivity)}</td>
      <td>${pct(m.specificity)}</td>
    </tr>`;
  }
  document.getElementById(containerId).innerHTML = `
    <table class="metrics-table">
      <thead>
        <tr><th></th><th>Accuracy</th><th>Sensitivity</th><th>Specificity</th></tr>
      </thead>
      <tbody>
        ${row('Train', trainM)}
        ${row('Test',  testM)}
      </tbody>
    </table>
    <div class="metrics-footnote">
      Positive class = AI (ChatGPT).&ensp;Sensitivity = TP/(TP+FN).&ensp;Specificity = TN/(TN+FP).
    </div>
  `;
}

// ── Charts ───────────────────────────────────────────────────────
function destroyChart(key) {
  if (charts[key]) { charts[key].destroy(); delete charts[key]; }
}

function renderTopWordsChart(canvasId, chartKey, topWords) {
  destroyChart(chartKey);
  const canvas = document.getElementById(canvasId);
  const { top, bottom } = topWords;
  const combined = [
    ...top.map(w => ({ word: w.word, score: w.score })),
    ...bottom.map(w => ({ word: w.word, score: w.score }))
  ];
  combined.sort((a, b) => Math.abs(b.score) - Math.abs(a.score));
  const display = combined.slice(0, top.length + bottom.length);
  const labels  = display.map(d => d.word);
  const values  = display.map(d => d.score);
  const colors  = values.map(v => v >= 0 ? 'rgba(74,144,217,.75)' : 'rgba(224,82,82,.75)');

  canvas.style.display = 'block';
  charts[chartKey] = new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [{ data: values, backgroundColor: colors,
        borderColor: colors.map(c => c.replace('.75', '1')), borderWidth: 1 }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => {
              const v = ctx.parsed.x;
              return `${v.toFixed(3)}  (${v >= 0 ? 'AI-leaning' : 'Human-leaning'})`;
            }
          }
        }
      },
      scales: { x: { title: { display: true, text: 'Log-odds / Weight' } } }
    }
  });
}

function renderEpochChart(canvasId, chartKey, epochTrainAccs, epochTestAccs) {
  destroyChart(chartKey);
  const canvas = document.getElementById(canvasId);
  canvas.style.display = 'block';
  const labels = epochTrainAccs.map((_, i) => `Epoch ${i + 1}`);
  charts[chartKey] = new Chart(canvas, {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'Train', data: epochTrainAccs.map(a => +(a * 100).toFixed(2)),
          fill: false, borderColor: '#4a90d9', backgroundColor: '#4a90d9',
          tension: 0.3, pointRadius: 4 },
        { label: 'Test',  data: epochTestAccs.map(a => +(a * 100).toFixed(2)),
          fill: false, borderColor: '#e05252', backgroundColor: '#e05252',
          borderDash: [5, 3], tension: 0.3, pointRadius: 4 }
      ]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: true, position: 'bottom' } },
      scales: {
        y: { min: 0, max: 100, title: { display: true, text: 'Accuracy (%)' } },
        x: { title: { display: true, text: 'Epoch' } }
      }
    }
  });
}

// ── Label chip helper ─────────────────────────────────────────────
function labelSpan(lbl) {
  const cls  = lbl === 0 ? 'label-human' : 'label-ai';
  const name = DATASET.label_names[lbl];
  return `<span class="${cls}">${name}</span>`;
}

// ── Examples view ─────────────────────────────────────────────────
function filterResults(results, filter) {
  switch (filter) {
    case 'correct':      return results.filter(r => r.correct);
    case 'wrong':        return results.filter(r => !r.correct);
    case 'conf-correct': {
      const ok = results.filter(r => r.correct).slice();
      ok.sort((a, b) => b.confidence - a.confidence);
      return ok.slice(0, TOP_CONF_N);
    }
    case 'conf-wrong': {
      const wr = results.filter(r => !r.correct).slice();
      wr.sort((a, b) => b.confidence - a.confidence);
      return wr.slice(0, TOP_CONF_N);
    }
    default: return results;
  }
}

// ── N-gram filter helpers ─────────────────────────────────────────
function textContainsFeature(text, feature) {
  const tokens = tokenize(text);
  if (!feature.includes('_')) return tokens.includes(feature);
  for (let i = 0; i < tokens.length - 1; i++)
    if (tokens[i] + '_' + tokens[i + 1] === feature) return true;
  return false;
}

function setNgramFilter(prefix, feature) {
  state[prefix].ngramFilter = feature;
  // Navigate to examples tab
  document.querySelector(`.inner-tab-btn[data-for="${prefix}"][data-panel="examples"]`).click();
  if (state[prefix].rawResults) {
    const testResults = applyThreshold(state[prefix].rawResults.test, state[prefix].threshold);
    renderExamples(prefix, testResults, state[prefix].filter);
  }
}

function renderExamples(prefix, results, filter) {
  const tbody    = document.getElementById(`${prefix}-examples-body`);
  const meta     = document.getElementById(`${prefix}-examples-meta`);
  const wrap     = document.getElementById(`${prefix}-examples-wrap`);
  const plc      = document.getElementById(`${prefix}-examples-placeholder`);
  const scoreHdr = document.getElementById(`${prefix}-score-hdr`);
  const filterBar = document.querySelector(`#${prefix}-panel-examples .filter-bar`);
  const banner   = document.getElementById(`${prefix}-ngram-banner`);
  const ngramFilter = state[prefix].ngramFilter;

  // Determine items to show and configure banner / filter-bar visibility
  let items;
  let metaHtml;
  if (ngramFilter) {
    filterBar.style.display = 'none';
    banner.style.display    = 'flex';
    document.getElementById(`${prefix}-ngram-chip`).textContent =
      ngramFilter.includes('_') ? ngramFilter.replace('_', ' ') : ngramFilter;

    const s      = state[prefix];
    const trainR = applyThreshold(s.rawResults.train, s.threshold);
    const testR  = applyThreshold(s.rawResults.test,  s.threshold);
    items = [
      ...trainR.filter(r => textContainsFeature(r.text, ngramFilter)).map(r => ({...r, split: 'train'})),
      ...testR.filter(r  => textContainsFeature(r.text,  ngramFilter)).map(r => ({...r, split: 'test'})),
    ];
    const nTrain = items.filter(r => r.split === 'train').length;
    const nTest  = items.filter(r => r.split === 'test').length;
    metaHtml = `<span class="meta-showing">
      <strong>${items.length}</strong> document${items.length !== 1 ? 's' : ''} contain
      &ldquo;${escHtml(ngramFilter.includes('_') ? ngramFilter.replace('_', ' ') : ngramFilter)}&rdquo;
      &ensp;<span class="meta-total">${nTrain} train</span>
      <span class="meta-total">${nTest} test</span>
    </span>`;
  } else {
    filterBar.style.display = '';
    banner.style.display    = 'none';
    items = filterResults(results, filter);
    const nTotal   = results.length;
    const nWrong   = results.filter(r => !r.correct).length;
    const nCorrect = nTotal - nWrong;
    const filterLabels = {
      all:            `all ${nTotal}`,
      correct:        `${nCorrect} correct`,
      wrong:          `${nWrong} wrong`,
      'conf-correct': `top ${TOP_CONF_N} most confident + correct`,
      'conf-wrong':   `top ${TOP_CONF_N} most confident + wrong`,
    };
    metaHtml = `
      <span class="meta-showing">Showing <strong>${items.length}</strong> items
        (${filterLabels[filter] || filter})</span>
      <span class="meta-counts">
        <span class="meta-total">${nTotal} total</span>
        <span class="meta-correct">&#10003; ${nCorrect} correct</span>
        <span class="meta-wrong">&#10007; ${nWrong} wrong</span>
      </span>`;
  }

  const isNB = (ngramFilter ? items : results).length > 0
    && (ngramFilter ? items : results)[0].isNB;
  if (scoreHdr) scoreHdr.textContent = isNB ? 'Log-odds' : 'Score';
  meta.innerHTML = metaHtml;

  tbody.innerHTML = '';
  items.forEach((r, idx) => {
    const cleanText = normalizeText(r.text);
    const flatText  = cleanText.replace(/\s+/g, ' ').trim();
    const snippet   = flatText.length > 90 ? flatText.slice(0, 90) + '…' : flatText;
    const scoreStr  = isNB ? r.rawScore.toFixed(2) : r.rawScore.toFixed(3);
    const detailId  = `${prefix}-detail-${idx}`;
    const splitBadge = r.split
      ? `<span class="split-badge split-${r.split}">${r.split}</span>`
      : '';

    // Data row
    const tr = document.createElement('tr');
    tr.className = r.correct ? 'row-ok' : 'row-err';
    tr.innerHTML = `
      <td class="col-num">${idx + 1}</td>
      <td class="col-text">
        ${splitBadge}<span class="ex-snippet">${escHtml(snippet)}</span>
        <button class="expand-btn" data-target="${detailId}" aria-label="Expand full text">&#9660;</button>
      </td>
      <td>${labelSpan(r.trueLabel)}</td>
      <td>${labelSpan(r.predLabel)}</td>
      <td class="col-score ${r.correct ? 'correct' : 'incorrect'}">${scoreStr}</td>
      <td class="col-check ${r.correct ? 'correct' : 'incorrect'}">${r.correct ? '&#10003;' : '&#10007;'}</td>
    `;
    tbody.appendChild(tr);

    // Detail row (hidden) — full text with token highlights rendered lazily on first expand
    const detail = document.createElement('tr');
    detail.id        = detailId;
    detail.className = 'detail-row';
    detail.style.display = 'none';
    detail.innerHTML = `<td colspan="6" class="detail-cell">
      <div class="detail-labels">
        ${r.split ? `<span class="split-badge split-${r.split}">${r.split}</span>&ensp;` : ''}
        True: ${labelSpan(r.trueLabel)}&ensp;Predicted: ${labelSpan(r.predLabel)}&ensp;
        ${isNB ? `Log-odds: <strong>${r.rawScore.toFixed(3)}</strong>` : `Score: <strong>${r.rawScore.toFixed(3)}</strong>`}
        &ensp;<span class="hl-legend">
          <mark style="background:rgba(74,144,217,.5);border-radius:2px;padding:0 2px">AI signal</mark>
          <mark style="background:rgba(224,82,82,.5);border-radius:2px;padding:0 2px">Human signal</mark>
          — top 40 tokens
        </span>
      </div>
      <div class="full-text"></div>
    </td>`;
    // Compute highlighted HTML only when the row is first opened
    let hlDone = false;
    detail._renderHL = () => {
      if (hlDone) return;
      hlDone = true;
      detail.querySelector('.full-text').innerHTML = buildHighlightedText(cleanText, prefix, 40);
    };
    tbody.appendChild(detail);
  });

  // Expand/collapse
  tbody.querySelectorAll('.expand-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const target  = document.getElementById(btn.dataset.target);
      const showing = target.style.display !== 'none';
      target.style.display = showing ? 'none' : 'table-row';
      btn.innerHTML = showing ? '&#9660;' : '&#9650;';
      if (!showing) target._renderHL?.();
    });
  });

  plc.style.display  = 'none';
  wrap.style.display = 'block';
}

function escHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// Convert literal \n / \t escape sequences that appear in some source texts
function normalizeText(t) {
  return t.replace(/\\n/g, '\n').replace(/\\t/g, '\t');
}

// Return the full text as HTML with the top-N most impactful tokens highlighted.
// In bigram mode, consecutive word pairs matching a top bigram feature are highlighted together.
function buildHighlightedText(text, prefix, topN = 40) {
  const model = state[prefix].model;
  if (!model) return escHtml(text);
  const isNB  = model instanceof NaiveBayes;
  const mode  = state[prefix].mode;

  const normTok = s => s.toLowerCase().replace(/[\u2018\u2019]/g, "'");

  const scoreFeature = f => {
    if (isNB) {
      const ll1 = model.logLik[1][f], ll0 = model.logLik[0][f];
      return (ll1 !== undefined && ll0 !== undefined) ? ll1 - ll0 : undefined;
    }
    return model.weights[f];  // undefined if absent
  };

  // Collect scores for all features present in this text
  const featureScores = {};
  const tokens = tokenize(text);  // already lowercased + normalised

  for (const tok of new Set(tokens)) {
    const s = scoreFeature(tok);
    if (s !== undefined) featureScores[tok] = s;
  }
  if (mode === 'bigram') {
    const seen = new Set();
    for (let i = 0; i < tokens.length - 1; i++) {
      const bg = tokens[i] + '_' + tokens[i + 1];
      if (!seen.has(bg)) {
        seen.add(bg);
        const s = scoreFeature(bg);
        if (s !== undefined) featureScores[bg] = s;
      }
    }
  }

  const sorted = Object.entries(featureScores)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const topMap = new Map(sorted.slice(0, topN));
  if (topMap.size === 0) return escHtml(text);
  const maxAbs = Math.abs(sorted[0][1]);

  const markHtml = (span, score, label) => {
    const intensity = maxAbs > 0 ? Math.abs(score) / maxAbs : 0;
    const alpha     = (0.22 + 0.55 * intensity).toFixed(2);
    const [r, g, b] = score > 0 ? [74, 144, 217] : [224, 82, 82];
    const dir       = score > 0 ? 'AI' : 'Human';
    return `<mark style="background:rgba(${r},${g},${b},${alpha});border-radius:2px;padding:0 1px"` +
           ` title="${escHtml(label)}: ${score.toFixed(3)} (${dir}-leaning)">${escHtml(span)}</mark>`;
  };

  if (mode !== 'bigram') {
    // Unigram path — original behaviour
    return text.split(/(\w+)/).map(part => {
      const lower = normTok(part);
      if (topMap.has(lower)) return markHtml(part, topMap.get(lower), lower);
      return escHtml(part);
    }).join('');
  }

  // Bigram-aware path: locate each token's character span in the original text,
  // then greedily highlight bigrams (preferred) or unigrams.
  const tokenMatches = [...text.matchAll(/[a-z0-9]+(?:'[a-z]+)*/gi)];
  const parts = [];
  let pos = 0;
  let i   = 0;
  while (i < tokenMatches.length) {
    const m     = tokenMatches[i];
    const lower = normTok(m[0]);
    let highlight = null;

    // Bigrams always beat unigrams when present in the top map (highest n wins).
    // Both constituent tokens are consumed so neither is re-highlighted individually.
    if (i + 1 < tokenMatches.length) {
      const bg      = lower + '_' + normTok(tokenMatches[i + 1][0]);
      const bgScore = topMap.get(bg);
      if (bgScore !== undefined) {
        highlight = {
          start: m.index,
          end:   tokenMatches[i + 1].index + tokenMatches[i + 1][0].length,
          score: bgScore,
          label: bg.replace('_', ' '),
        };
        i += 2;
      }
    }

    if (!highlight) {
      const uniScore = topMap.get(lower);
      if (uniScore !== undefined) {
        highlight = { start: m.index, end: m.index + m[0].length, score: uniScore, label: lower };
      }
      i++;
    }

    if (highlight) {
      parts.push(escHtml(text.slice(pos, highlight.start)));
      parts.push(markHtml(text.slice(highlight.start, highlight.end), highlight.score, highlight.label));
      pos = highlight.end;
    }
  }
  parts.push(escHtml(text.slice(pos)));
  return parts.join('');
}

// ── Word-weights tab ──────────────────────────────────────────────
const WEIGHTS_DISPLAY_MAX = 200;

function renderWeightsTab(prefix) {
  const s = state[prefix];
  if (!s.wordScores) return;
  const isNB = s.model instanceof NaiveBayes;

  const searchVal = (document.getElementById(`${prefix}-weights-search`) || {}).value || '';
  const dirFilter = s.weightsDirFilter;

  let items = s.wordScores;
  if (dirFilter === 'ai')    items = items.filter(w => w.score > 0);
  if (dirFilter === 'human') items = items.filter(w => w.score < 0);
  if (searchVal.trim()) {
    const q = searchVal.trim().toLowerCase();
    items = items.filter(w => w.word.includes(q));
  }

  const total   = items.length;
  const display = items.slice(0, WEIGHTS_DISPLAY_MAX);
  // s.wordScores is sorted by |score| desc, so index 0 has max
  const maxAbs  = s.wordScores.length > 0 ? Math.abs(s.wordScores[0].score) : 1;

  const scoreHdrEl = document.getElementById(`${prefix}-weights-score-hdr`);
  if (scoreHdrEl) scoreHdrEl.textContent = isNB ? 'Log-odds' : 'Weight';

  const metaEl = document.getElementById(`${prefix}-weights-meta`);
  metaEl.textContent = `Showing ${display.length} of ${total} word${total !== 1 ? 's' : ''}`
    + (total < s.wordScores.length ? ` (filtered from ${s.wordScores.length})` : '');

  const tbody = document.getElementById(`${prefix}-weights-body`);
  tbody.innerHTML = '';
  display.forEach((w, i) => {
    const barPct  = maxAbs > 0 ? Math.min(100, (Math.abs(w.score) / maxAbs) * 100) : 0;
    const cls     = w.score > 0 ? 'ai' : 'human';
    const signStr = w.score > 0 ? '+' : '';
    const tr = document.createElement('tr');
    tr.className = 'weight-row-clickable';
    tr.title     = `Click to show documents containing "${w.word}"`;
    tr.innerHTML = `
      <td class="wt-rank">${i + 1}</td>
      <td class="wt-word">${escHtml(w.word)}</td>
      <td class="wt-score ${w.score > 0 ? 'correct' : 'incorrect'}">${signStr}${w.score.toFixed(3)}</td>
      <td class="wt-bar">
        <div class="bar-track">
          <div class="bar-fill ${cls}" style="width:${barPct.toFixed(1)}%"></div>
        </div>
      </td>
    `;
    tr.addEventListener('click', () => setNgramFilter(prefix, w.word));
    tbody.appendChild(tr);
  });

  document.getElementById(`${prefix}-weights-placeholder`).style.display = 'none';
  document.getElementById(`${prefix}-weights-wrap`).style.display = 'block';
}

function initWeightsTabs() {
  for (const prefix of ['nb', 'perc']) {
    const searchEl = document.getElementById(`${prefix}-weights-search`);
    if (searchEl) searchEl.addEventListener('input', () => renderWeightsTab(prefix));

    document.querySelectorAll(`.dir-btn[data-prefix="${prefix}"]`).forEach(btn => {
      btn.addEventListener('click', () => {
        state[prefix].weightsDirFilter = btn.dataset.dir;
        btn.closest('.dir-toggle').querySelectorAll('.dir-btn')
           .forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        renderWeightsTab(prefix);
      });
    });
  }
}

// ── Threshold update (no retraining) ─────────────────────────────
function onThresholdChange(prefix) {
  const s = state[prefix];
  if (!s.rawResults) return;
  const trainResults = applyThreshold(s.rawResults.train, s.threshold);
  const testResults  = applyThreshold(s.rawResults.test,  s.threshold);
  renderMetrics(`${prefix}-accuracy-display`, trainResults, testResults);
  renderExamples(prefix, testResults, s.filter);
}

// ── N-gram filter clear button wiring ────────────────────────────
function initNgramClear() {
  document.querySelectorAll('.ngram-clear').forEach(btn => {
    btn.addEventListener('click', () => {
      const prefix = btn.dataset.prefix;
      state[prefix].ngramFilter = null;
      if (state[prefix].rawResults) {
        const testResults = applyThreshold(state[prefix].rawResults.test, state[prefix].threshold);
        renderExamples(prefix, testResults, state[prefix].filter);
      }
    });
  });
}

// ── Filter button wiring ──────────────────────────────────────────
function initFilterBtns() {
  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const prefix = btn.dataset.model;
      const filter = btn.dataset.filter;
      state[prefix].filter = filter;
      btn.closest('.filter-bar').querySelectorAll('.filter-btn')
         .forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      if (state[prefix].rawResults) {
        const testResults = applyThreshold(state[prefix].rawResults.test, state[prefix].threshold);
        renderExamples(prefix, testResults, filter);
      }
    });
  });
}

// ── Inner tab switching ───────────────────────────────────────────
function initInnerTabs() {
  document.querySelectorAll('.inner-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const prefix = btn.dataset.for;
      const panel  = btn.dataset.panel;
      btn.closest('.inner-tab-bar').querySelectorAll('.inner-tab-btn')
         .forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      btn.closest('.inner-pane').querySelectorAll('.inner-panel')
         .forEach(p => p.classList.remove('active'));
      document.getElementById(`${prefix}-panel-${panel}`).classList.add('active');
    });
  });
}

// ── Outer tab switching ───────────────────────────────────────────
function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
    });
  });
}

// ── Status bar helpers ────────────────────────────────────────────
function setStatus(id, msg, loading = false) {
  document.getElementById(id).innerHTML = loading
    ? `<span class="spinner"></span>${msg}`
    : msg;
}

// ── Log-scale slider helper ───────────────────────────────────────
function logSliderValue(v, min, max) {
  const minLog = Math.log10(min), maxLog = Math.log10(max);
  return Math.pow(10, minLog + (v / 100) * (maxLog - minLog));
}

// ── Slider bindings ───────────────────────────────────────────────
function bindSliders() {
  function bind(id, valId, fn) {
    const el  = document.getElementById(id);
    const vel = document.getElementById(valId);
    el.addEventListener('input', () => { vel.textContent = fn(el); });
    vel.textContent = fn(el);   // init
  }

  bind('nb-train-size',   'nb-train-size-val',   el => el.value);
  bind('perc-train-size', 'perc-train-size-val', el => el.value);
  bind('perc-epochs',     'perc-epochs-val',     el => el.value);
  bind('perc-lr',         'perc-lr-val',  el => logSliderValue(+el.value, 0.001, 1.0).toFixed(3));

  // NB threshold
  const nbThresh = document.getElementById('nb-thresh');
  const nbThreshVal = document.getElementById('nb-thresh-val');
  nbThresh.addEventListener('input', () => {
    const v = NB_THRESH_VALUES[+nbThresh.value];
    nbThreshVal.value = v.toFixed(2);
    state.nb.threshold = v;
    onThresholdChange('nb');
  });
  nbThreshVal.value = NB_THRESH_VALUES[+nbThresh.value].toFixed(2);
  nbThreshVal.addEventListener('keydown', e => { if (e.key === 'Enter') nbThreshVal.blur(); });
  nbThreshVal.addEventListener('blur', () => {
    const parsed = parseFloat(nbThreshVal.value);
    if (!isFinite(parsed)) {
      nbThreshVal.value = state.nb.threshold.toFixed(2);
      return;
    }
    state.nb.threshold = parsed;
    nbThreshVal.value = parsed.toFixed(2);
    onThresholdChange('nb');
  });

  // Perceptron threshold
  const percThresh = document.getElementById('perc-thresh');
  const percThreshVal = document.getElementById('perc-thresh-val');
  percThresh.addEventListener('input', () => {
    const v = PERC_THRESH_VALUES[+percThresh.value];
    percThreshVal.value = v.toFixed(2);
    state.perc.threshold = v;
    onThresholdChange('perc');
  });
  percThreshVal.value = PERC_THRESH_VALUES[+percThresh.value].toFixed(2);
  percThreshVal.addEventListener('keydown', e => { if (e.key === 'Enter') percThreshVal.blur(); });
  percThreshVal.addEventListener('blur', () => {
    const parsed = parseFloat(percThreshVal.value);
    if (!isFinite(parsed)) {
      percThreshVal.value = state.perc.threshold.toFixed(2);
      return;
    }
    state.perc.threshold = parsed;
    percThreshVal.value = parsed.toFixed(2);
    onThresholdChange('perc');
  });
}

// ── Live inference ────────────────────────────────────────────────
function setupInference(prefix, getModel, getMode) {
  const btn = document.getElementById(`${prefix}-infer-btn`);
  btn.addEventListener('click', () => {
    const model = getModel();
    if (!model) return;
    const text = document.getElementById(`${prefix}-infer-text`).value.trim();
    if (!text) return;

    const mode      = getMode();
    model._mode     = mode;
    const feats     = extractFeatures(text, mode);
    const raw       = model.rawScore(feats);
    const isNB      = raw.odds !== undefined;
    const rawVal    = isNB ? raw.odds : raw.score;
    const threshold = state[prefix].threshold;
    const lbl       = rawVal >= threshold ? 1 : 0;

    // Top contributing words
    let topContrib = [];
    const seen = new Set();
    for (const tok of tokenize(text)) {
      if (seen.has(tok)) continue;
      seen.add(tok);
      if (isNB) {
        const ll  = model.logLik[1][tok];
        const opp = model.logLik[0][tok];
        if (ll !== undefined && opp !== undefined)
          topContrib.push({ word: tok, score: ll - opp });
      } else {
        const w = model.weights[tok];
        if (w) topContrib.push({ word: tok, score: w });
      }
    }
    topContrib.sort((a, b) => Math.abs(b.score) - Math.abs(a.score));
    topContrib = topContrib.slice(0, 8);

    const cls        = lbl === 0 ? 'human' : 'ai';
    const verdict    = DATASET.label_names[lbl];
    const scoreLabel = isNB ? `Odds: ${rawVal.toFixed(2)}` : `Score: ${rawVal.toFixed(3)}`;

    let wordsHtml = '';
    if (topContrib.length > 0) {
      wordsHtml = '<div class="top-words-list">Top contributing words: ' +
        topContrib.map(w => {
          const wCls = w.score > 0 ? 'tw-ai' : 'tw-human';
          const disp = isNB ? Math.exp(w.score).toFixed(2) : w.score.toFixed(2);
          return `<span class="${wCls}">${w.word} (${disp})</span>`;
        }).join('') + '</div>';
    }

    document.getElementById(`${prefix}-infer-result`).innerHTML = `
      <div class="inference-result ${cls}">
        <div class="verdict">Prediction: ${verdict}</div>
        <div class="score-line">${scoreLabel}
          &ensp;(${lbl === 1 ? 'AI-leaning' : 'Human-leaning'})</div>
        ${wordsHtml}
      </div>
    `;
  });
}

// ── NB training flow ─────────────────────────────────────────────
let nbModel = null;
let nbMode  = 'unigram';

function initNaiveBayes() {
  const btn = document.getElementById('nb-train-btn');
  btn.addEventListener('click', async () => {
    if (!TRAIN_ALL) { setStatus('nb-status', 'Data not loaded yet.'); return; }
    btn.disabled = true;
    setStatus('nb-status', 'Training…', true);
    await tick();

    try {
      const trainSize     = +document.getElementById('nb-train-size').value;
      const mode          = document.querySelector('input[name="nb-feat"]:checked').value;
      nbMode = state.nb.mode = mode;
      const trainExamples = TRAIN_ALL.slice(0, trainSize);
      const testExamples  = TEST_ALL;

      setStatus('nb-status', 'Building vocabulary…', true);
      await tick();
      const vocab = buildVocab(trainExamples, mode);

      setStatus('nb-status', 'Training Naive Bayes…', true);
      await tick();
      nbModel = new NaiveBayes();
      nbModel.train(trainExamples, vocab, mode, 1);
      state.nb.model = nbModel;

      setStatus('nb-status', 'Evaluating…', true);
      await tick();

      const trainRaw = computeRawResults(trainExamples, nbModel, mode);
      const testRaw  = computeRawResults(testExamples,  nbModel, mode);
      state.nb.rawResults = { train: trainRaw, test: testRaw };

      const thresh       = state.nb.threshold;
      const trainResults = applyThreshold(trainRaw, thresh);
      const testResults  = applyThreshold(testRaw,  thresh);

      setStatus('nb-status', 'Rendering…', true);
      await tick();

      renderMetrics('nb-accuracy-display', trainResults, testResults);

      const tw = nbModel.topWords(10);
      document.getElementById('nb-topwords-placeholder').style.display = 'none';
      renderTopWordsChart('nb-topwords-chart', 'nb-topwords', tw);

      state.nb.ngramFilter = null;
      renderExamples('nb', testResults, state.nb.filter);

      // Word scores for the weights tab (all vocab words, sorted by |log-odds|)
      state.nb.wordScores = nbModel.vocabList
        .map(f => ({ word: f, score: (nbModel.logLik[1][f] || 0) - (nbModel.logLik[0][f] || 0) }))
        .sort((a, b) => Math.abs(b.score) - Math.abs(a.score));
      renderWeightsTab('nb');

      document.getElementById('nb-thresh').disabled          = false;
      document.getElementById('nb-weights-search').disabled  = false;
      document.getElementById('nb-infer-text').disabled      = false;
      document.getElementById('nb-infer-btn').disabled       = false;
      document.getElementById('nb-infer-result').innerHTML = '';

      const acc = metricsFromResults(testResults).accuracy;
      setStatus('nb-status', `Done. Test accuracy: ${(acc * 100).toFixed(1)}%`);
    } catch (e) {
      console.error(e);
      setStatus('nb-status', 'Error: ' + e.message);
    } finally {
      btn.disabled = false;
    }
  });
}

// ── Perceptron training flow ──────────────────────────────────────
let percModel = null;
let percMode  = 'unigram';

function initPerceptron() {
  const btn = document.getElementById('perc-train-btn');
  btn.addEventListener('click', async () => {
    if (!TRAIN_ALL) { setStatus('perc-status', 'Data not loaded yet.'); return; }
    btn.disabled = true;
    setStatus('perc-status', 'Training…', true);
    await tick();

    try {
      const trainSize     = +document.getElementById('perc-train-size').value;
      const epochs        = +document.getElementById('perc-epochs').value;
      const lr            = logSliderValue(+document.getElementById('perc-lr').value, 0.001, 1.0);
      const mode          = document.querySelector('input[name="perc-feat"]:checked').value;
      percMode = state.perc.mode = mode;
      const trainExamples = TRAIN_ALL.slice(0, trainSize);
      const testExamples  = TEST_ALL;

      setStatus('perc-status', 'Training Perceptron…', true);
      await tick();
      percModel = new Perceptron();
      const { epochTrainAccs, epochTestAccs } =
        percModel.train(trainExamples, testExamples, epochs, lr, mode);
      state.perc.model = percModel;

      setStatus('perc-status', 'Evaluating…', true);
      await tick();

      const trainRaw = computeRawResults(trainExamples, percModel, mode);
      const testRaw  = computeRawResults(testExamples,  percModel, mode);
      state.perc.rawResults = { train: trainRaw, test: testRaw };

      const thresh       = state.perc.threshold;
      const trainResults = applyThreshold(trainRaw, thresh);
      const testResults  = applyThreshold(testRaw,  thresh);

      setStatus('perc-status', 'Rendering…', true);
      await tick();

      renderMetrics('perc-accuracy-display', trainResults, testResults);

      const tw = percModel.topWords(10);
      document.getElementById('perc-topwords-placeholder').style.display = 'none';
      renderTopWordsChart('perc-topwords-chart', 'perc-topwords', tw);

      document.getElementById('perc-epoch-placeholder').style.display = 'none';
      renderEpochChart('perc-epoch-chart', 'perc-epoch', epochTrainAccs, epochTestAccs);

      state.perc.ngramFilter = null;
      renderExamples('perc', testResults, state.perc.filter);

      // Word scores for the weights tab (all learned weights, sorted by |weight|)
      state.perc.wordScores = Object.entries(percModel.weights)
        .map(([word, score]) => ({ word, score }))
        .sort((a, b) => Math.abs(b.score) - Math.abs(a.score));
      renderWeightsTab('perc');

      document.getElementById('perc-thresh').disabled         = false;
      document.getElementById('perc-weights-search').disabled = false;
      document.getElementById('perc-infer-text').disabled     = false;
      document.getElementById('perc-infer-btn').disabled      = false;
      document.getElementById('perc-infer-result').innerHTML = '';

      const acc = metricsFromResults(testResults).accuracy;
      setStatus('perc-status', `Done. Test accuracy: ${(acc * 100).toFixed(1)}%`);
    } catch (e) {
      console.error(e);
      setStatus('perc-status', 'Error: ' + e.message);
    } finally {
      btn.disabled = false;
    }
  });
}

// ── Utility ──────────────────────────────────────────────────────
function tick() { return new Promise(r => setTimeout(r, 10)); }

// ── Bootstrap ────────────────────────────────────────────────────
async function init() {
  initTabs();
  initInnerTabs();
  initFilterBtns();
  initNgramClear();
  initWeightsTabs();
  bindSliders();

  setStatus('nb-status',   'Loading data…', true);
  setStatus('perc-status', 'Loading data…', true);

  try {
    await loadData();
    const msg = `Data loaded: ${TRAIN_ALL.length} train, ${TEST_ALL.length} test. Ready.`;
    setStatus('nb-status',   msg);
    setStatus('perc-status', msg);
  } catch (e) {
    const msg = `Failed to load data: ${e.message}. Serve with: python -m http.server`;
    setStatus('nb-status',   msg);
    setStatus('perc-status', msg);
    return;
  }

  initNaiveBayes();
  initPerceptron();
  setupInference('nb',   () => nbModel,   () => nbMode);
  setupInference('perc', () => percModel, () => percMode);
}

document.addEventListener('DOMContentLoaded', init);
