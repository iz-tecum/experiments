// rank.js
// Loads a trained pairwise ranker (rank_model.json) and produces:
// - raw score = w · x + bias
// - percentile-mapped 0–10 score within a pool (optional)

(function (global) {
  "use strict";

  const MODEL_URL = "./model/rank_model.json";

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

    // upper-bound index: count strictly below raw
    let idx = 0;
    while (idx < sorted.length && sorted[idx] < raw) idx++;

    const rank = idx;
    const pct = sorted.length <= 1 ? 0.5 : rank / (sorted.length - 1);

    return round1(pct * 10);
  }

  function assertModelCompatible(model) {
    if (!model || !Array.isArray(model.weights)) {
      throw new Error("Invalid model JSON (missing weights array)");
    }

    // feature_version check (if PMEFeatures is present)
    const fvModel = model.feature_version;
    const fvLocal = global.PMEFeatures ? global.PMEFeatures.FEATURE_VERSION : null;

    if (fvLocal != null && fvModel != null && Number(fvModel) !== Number(fvLocal)) {
      throw new Error(
        `Model/feature version mismatch: model=${fvModel}, features=${fvLocal}. Re-train or deploy matching files.`
      );
    }

    // Basic sanity: weights numeric
    for (let i = 0; i < model.weights.length; i++) {
      const v = Number(model.weights[i]);
      if (!Number.isFinite(v)) throw new Error(`Invalid weight at index ${i}`);
      model.weights[i] = v;
    }

    // bias optional
    model.bias = Number.isFinite(Number(model.bias)) ? Number(model.bias) : 0.0;

    return model;
  }

  async function loadModel(url = MODEL_URL) {
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) throw new Error(`Could not load model: ${r.status}`);
    const j = await r.json();
    return assertModelCompatible(j);
  }

  // raw score for a single applicant
  function scoreRaw(model, features) {
    if (!model || !Array.isArray(model.weights)) throw new Error("Model not loaded");
    if (!Array.isArray(features)) throw new Error("Features must be an array");

    if (model.weights.length !== features.length) {
      throw new Error(
        `Feature length mismatch: model expects ${model.weights.length}, got ${features.length}`
      );
    }

    return dot(model.weights, features) + (model.bias || 0.0);
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