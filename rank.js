// rank.js
// Loads a trained pairwise ranker (rank_model.json) and produces:
// - raw score = w · x
// - percentile-mapped 0–10 score within a pool (optional)

(function (global) {
  "use strict";

  const MODEL_URL = "./public/model/rank_model.json"; // works when index.html is at repo root

  function dot(w, x) {
    let s = 0;
    const n = Math.min(w.length, x.length);
    for (let i = 0; i < n; i++) s += w[i] * x[i];
    return s;
  }

  function round1(x) {
    return Math.round(x * 10) / 10;
  }

  function percentileScore(raw, rawAll) {
    if (!rawAll || rawAll.length === 0) return 5.0;
    const sorted = [...rawAll].sort((a, b) => a - b);
    // upper-bound index
    let idx = 0;
    while (idx < sorted.length && sorted[idx] < raw) idx++;
    const rank = idx; // number below raw
    const pct = sorted.length <= 1 ? 0.5 : rank / (sorted.length - 1);
    return round1(pct * 10);
  }

  async function loadModel(url = MODEL_URL) {
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) throw new Error(`Could not load model: ${r.status}`);
    const j = await r.json();
    if (!j || !Array.isArray(j.weights)) throw new Error("Invalid model JSON (missing weights)");
    return j;
  }

  // raw score for a single applicant
  function scoreRaw(model, features) {
    return dot(model.weights, features);
  }

  // returns { raw, score_0_10 } using a pool of raw scores (optional)
  function scoreWithPool(model, features, poolRawScores) {
    const raw = scoreRaw(model, features);
    const score_0_10 = percentileScore(raw, poolRawScores || [raw]);
    return { raw, score_0_10 };
  }

  global.PMERanker = {
    MODEL_URL,
    loadModel,
    scoreRaw,
    scoreWithPool,
    percentileScore
  };

})(window);