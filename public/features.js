// features.js
// Shared deterministic feature extraction for PME learning-to-rank (budget $0).
// Used by: tools/feature_extractor.html, index.html (applicant page), and any future tooling.
//
// NOTE (honest + important):
// - No feature function can "perfectly" score applicants. What we can do is build a stable,
//   mathematically well-behaved signal map (bounded, smooth, hard-to-game), and then let your
//   pairwise ranker learn the weights from your chapter’s judgments.
// - We *will* output features to 6 decimal places for stable training.

(function (global) {
  "use strict";

  // ==========
  // Feature schema
  // ==========
  const FEATURE_VERSION = 2;

  // IMPORTANT: Keep order stable once you collect comparisons.
  // We keep the same 21 slots, but improve what they measure (esp. essays + upper-math).
  const FEATURE_NAMES = [
    "gpa_score_0_10",
    "calc12_binary_0_or_10",
    "upper_math_score_0_10",

    "resume_len_log",
    "resume_has_education",
    "resume_has_experience",
    "resume_has_projects",
    "resume_has_skills",

    "kw_math_engagement",
    "kw_research_exposition",
    "kw_teaching_mentoring",
    "kw_leadership_service",
    "kw_awards_honors",
    "kw_competitions",

    // Reinterpreted: these are NO LONGER "length rewards".
    // They become "2–4 sentence bandpass quality" signals.
    "essay_math_len_log",
    "essay_comm_len_log",

    // Reinterpreted to reflect the new prompts:
    // - "essay_math_reasoning_markers" becomes a "personal arc / reflection" score
    // - "essay_comm_specificity_markers" becomes a "math+contribution plan specificity" score
    "essay_math_reasoning_markers",
    "essay_comm_specificity_markers",

    // Still useful: math terms presence in each answer
    "essay_math_math_terms",
    "essay_comm_math_terms",

    "keyword_density_penalty"
  ];

  // ==========
  // Keyword buckets (small + interpretable)
  // ==========
  const KW = {
    math_engagement: [
      "problem session",
      "seminar",
      "colloquium",
      "reading group",
      "math circle",
      "putnam",
      "olympiad",
      "contest",
      "proof",
      "theorem",
      "lemma"
    ],
    research_exposition: [
      "research",
      "paper",
      "preprint",
      "poster",
      "presentation",
      "talk",
      "expository",
      "publication",
      "manuscript",
      "arxiv",
      "journal"
    ],
    teaching_mentoring: [
      "ta",
      "teaching assistant",
      "tutor",
      "tutoring",
      "mentor",
      "mentoring",
      "instructor"
    ],
    leadership_service: [
      "president",
      "founder",
      "cofounder",
      "chair",
      "director",
      "lead",
      "captain",
      "organize",
      "organized",
      "organizing",
      "outreach",
      "volunteer",
      "service"
    ],
    awards_honors: [
      "award",
      "honor",
      "honours",
      "scholarship",
      "fellowship",
      "recipient",
      "prize",
      "winner",
      "finalist"
    ],
    competitions: [
      "putnam",
      "imo",
      "usamo",
      "amc",
      "aime",
      "icpc",
      "hackathon",
      "math competition",
      "contest"
    ],
    math_terms: [
      "proof",
      "theorem",
      "lemma",
      "corollary",
      "define",
      "assume",
      "therefore",
      "hence",
      "group",
      "ring",
      "field",
      "analysis",
      "topology",
      "algebra",
      "complex",
      "measure",
      "probability",
      "linear",
      "eigen",
      "integral",
      "series",
      "converge",
      "holomorphic"
    ],
    reasoning_markers: [
      "assume",
      "suppose",
      "let ",
      "then",
      "therefore",
      "hence",
      "thus",
      "it follows",
      "we claim",
      "consider"
    ],
    specificity_markers: [
      "talk",
      "problem session",
      "reading group",
      "workshop",
      "weekly",
      "biweekly",
      "invite",
      "speaker",
      "panel",
      "office hours",
      "study jam",
      "problem set",
      "mini-talk",
      "minitalk",
      "chalk talk",
      "chalktalk",
      "rsvp"
    ],

    // New: narrative arc markers for the personal 2–4 sentence response
    personal_context_markers: [
      "i grew up",
      "i come from",
      "first-gen",
      "first generation",
      "financial",
      "bureaucratic",
      "stutter",
      "disability",
      "overwhelmed",
      "stress",
      "struggle",
      "difficult",
      "hard",
      "failure",
      "doubt"
    ],
    personal_action_markers: [
      "i started",
      "i learned",
      "i taught",
      "i practiced",
      "i worked",
      "i asked",
      "i joined",
      "i built",
      "i wrote",
      "i proved",
      "i kept",
      "i continued"
    ],
    personal_reflection_markers: [
      "i realized",
      "i learned that",
      "it taught me",
      "since then",
      "now i",
      "because of that",
      "as a result",
      "i understand",
      "i changed"
    ],

    // New: plan markers for math+contribution answer
    plan_markers: [
      "i would",
      "i will",
      "i plan",
      "i want to",
      "we could",
      "we will",
      "host",
      "run",
      "organize",
      "start",
      "lead",
      "facilitate",
      "schedule",
      "coordinate"
    ]
  };

  // ==========
  // Numeric helpers (smooth, bounded, stable)
  // ==========
  function clamp(x, a, b) {
    return Math.max(a, Math.min(b, x));
  }

  function safeNum(x, fallback = NaN) {
    const v = Number(x);
    return Number.isFinite(v) ? v : fallback;
  }

  function round6(x) {
    return Math.round(x * 1e6) / 1e6;
  }

  function sigmoid(z) {
    // stable logistic
    const t = clamp(z, -30, 30);
    return 1 / (1 + Math.exp(-t));
  }

  function softplus(z) {
    // stable softplus
    const t = clamp(z, -30, 30);
    return Math.log1p(Math.exp(t));
  }

  function satExp(x, alpha) {
    // concave saturating map: 0 -> 0, large -> 1
    // 1 - e^{-alpha x}
    return 1 - Math.exp(-alpha * Math.max(0, x));
  }

  function gaussianBand(x, mu, sigma) {
    const s = Math.max(1e-6, sigma);
    const u = (x - mu) / s;
    return Math.exp(-0.5 * u * u);
  }

  function logNormLen(s) {
    const n = (s || "").trim().length;
    // log(1+n) scaled roughly into [0,10] for typical resume lengths
    return clamp((Math.log1p(n) / Math.log(1 + 6000)) * 10, 0, 10);
  }

  // ==========
  // Text helpers
  // ==========
  function normText(s) {
    return String(s || "")
      .toLowerCase()
      .replace(/\s+/g, " ")
      .trim();
  }

  function containsAny(text, arr) {
    const t = normText(text);
    return arr.some((k) => t.includes(k));
  }

  function countHits(text, arr) {
    // "unique hit count": counts each keyword at most once
    const t = normText(text);
    let hits = 0;
    for (const k of arr) {
      if (!k) continue;
      if (t.includes(k)) hits += 1;
    }
    return hits;
  }

  function countSentences(text) {
    const t = String(text || "").trim();
    if (!t) return 0;
    // Split on sentence enders; keep simple and deterministic
    const parts = t.split(/[.!?]+/g).map((s) => s.trim()).filter(Boolean);
    return parts.length;
  }

  function wordCount(text) {
    const t = String(text || "").trim();
    if (!t) return 0;
    return t.split(/\s+/g).filter(Boolean).length;
  }

  // ==========
  // Anti-gaming penalty (learnable feature)
  // ==========
  function densityPenalty(text) {
    const t = normText(text);
    const n = t.length;
    if (n < 200) return 0;

    const totalHits =
      countHits(t, KW.math_engagement) +
      countHits(t, KW.research_exposition) +
      countHits(t, KW.teaching_mentoring) +
      countHits(t, KW.leadership_service) +
      countHits(t, KW.awards_honors) +
      countHits(t, KW.competitions) +
      countHits(t, KW.math_terms);

    // hits per 1000 chars
    const rate = totalHits / (n / 1000);

    // Smooth penalty: turns on around ~40 hits/1000, saturates toward 10.
    // Using softplus to avoid sharp discontinuities:
    const z = (rate - 40) / 8; // scale
    const p = softplus(z);     // ~0 for negative, grows for positive
    const scaled = 10 * (1 - Math.exp(-0.55 * p)); // saturate
    return clamp(scaled, 0, 10);
  }

  // ==========
  // Academic scoring (0–10)
  // ==========
  function scoreGpa(gpa) {
    // Columbia can go to ~4.33 due to A+
    const x = clamp(gpa, 0, 4.33);

    // Smooth, monotone, and "threshold-aware":
    // - Below 3.4: steeply lower but not zeroed out
    // - 3.4–3.9: strong ramp
    // - >=3.9: saturates at 10
    //
    // Use a logistic around 3.55 plus a refinement bump near 3.85.
    const a = sigmoid((x - 3.55) / 0.18); // main competitiveness
    const b = sigmoid((x - 3.85) / 0.10); // excellence bump
    const s = 10 * clamp(0.15 + 0.65 * a + 0.20 * b, 0, 1);

    // Hard cap at 10.
    return clamp(s, 0, 10);
  }

  function scoreCalc(val) {
    if (val === "yes") return 10;
    if (val === "no") return 0;
    return 0;
  }

  // Back-compat scoring (old UI used upperCount + upperRigor).
  function scoreUpperCourses(count, rigor) {
    const n = clamp(count || 0, 0, 20);

    // Concave saturation on course count (so you can't "win" by listing a ton)
    // normalized to 0..10
    const countScore = 10 * satExp(n, 0.45); // ~ 6 courses already near saturation

    // Rigor bump (bounded)
    let bonus = 0;
    if (rigor === "proof") bonus = 0.6;
    if (rigor === "upper") bonus = 1.2;
    if (rigor === "grad") bonus = 1.8;

    return clamp(countScore + bonus, 0, 10);
  }

  // New: compute upper-math score from marked course codes (max 6).
  // This is deterministic and does not require any external catalog at runtime.
  function scoreUpperFromCourses(courses) {
    const list = Array.isArray(courses) ? courses.slice(0, 6) : [];
    if (!list.length) return 0;

    // Points by level (monotone, with diminishing returns via saturation later)
    function pointsFor(codeRaw) {
      const code = String(codeRaw || "").trim().toUpperCase().replace(/\s+/g, " ");
      // Expected formats: "MATH UN3007", "MATH GU4061", "MATH GR6151", "MATH BC2006"
      const m = code.match(/^MATH\s+(UN|GU|GR|BC)\s?(\d{4})$/);
      if (!m) return 0;

      const lvl = m[1];
      const num = parseInt(m[2], 10);

      if (lvl === "BC") return 1.0;

      if (lvl === "UN") {
        // UN1xxx/2xxx/3xxx: interpret higher number as more advanced
        if (num >= 3000) return 1.35;
        if (num >= 2000) return 1.05;
        return 0.75;
      }

      if (lvl === "GU") {
        // GU4xxx
        return 1.70;
      }

      if (lvl === "GR") {
        // GR5xxx/6xxx
        if (num >= 6000) return 2.35;
        return 2.05;
      }

      return 0;
    }

    let pts = 0;
    for (const c of list) pts += pointsFor(c);

    // Saturating map to 0..10.
    // 4–6 solid advanced courses should land high, but never blow up.
    const s = 10 * satExp(pts, 0.55);
    return clamp(s, 0, 10);
  }

  // ==========
  // Essay scoring for 2–4 sentences (no length reward)
  // ==========
  function bandpassSentenceScore(text) {
    const s = countSentences(text);

    // Ideal: 3 sentences (2–4 near-max).
    // We use a Gaussian band (smooth) + a tiny floor for non-empty text.
    const base = gaussianBand(s, 3, 1.0); // 2–4 stays strong
    const nonEmpty = String(text || "").trim().length > 0 ? 1 : 0;

    // Map to 0..10 with slight floor if nonempty (so 1 sentence isn't crushed)
    const score = 10 * clamp(0.12 * nonEmpty + 0.88 * base, 0, 1);
    return clamp(score, 0, 10);
  }

  function personalArcScore(text) {
    // We want: (context/challenge) + (action) + (reflection) in 2–4 sentences.
    const t = normText(text);

    const c = containsAny(t, KW.personal_context_markers) ? 1 : 0;
    const a = containsAny(t, KW.personal_action_markers) ? 1 : 0;
    const r = containsAny(t, KW.personal_reflection_markers) ? 1 : 0;

    // Require at least two of three for high score, but keep smooth:
    const raw = (c + a + r) / 3; // 0..1

    // Sharpen around 2/3 via logistic
    const s = 10 * sigmoid((raw - 0.55) / 0.12);
    return clamp(s, 0, 10);
  }

  function planSpecificityScore(text) {
    // For math+contribution prompt: reward a concrete plan.
    const t = normText(text);

    const planHits = countHits(t, KW.plan_markers);          // unique plan verbs
    const specHits = countHits(t, KW.specificity_markers);   // concrete event markers

    // Diminishing returns: saturate quickly (2–4 hits is plenty)
    const plan = 10 * satExp(planHits, 0.9);
    const spec = 10 * satExp(specHits, 0.9);

    // Combine with emphasis on specificity (what happens in the room)
    const s = clamp(0.45 * plan + 0.55 * spec, 0, 10);
    return s;
  }

  function mathTermScore(text) {
    const t = normText(text);
    const hits = countHits(t, KW.math_terms);
    // saturate quickly: 6–8 unique terms is enough for max
    const s = 10 * satExp(hits, 0.42);
    return clamp(s, 0, 10);
  }

  function reasoningMarkerScore(text) {
    const t = normText(text);
    const hits = countHits(t, KW.reasoning_markers);
    // small signal: don't over-reward keywording; saturate hard
    const s = 10 * satExp(hits, 0.55);
    return clamp(s, 0, 10);
  }

  // ==========
  // Main: build feature vector
  // ==========
  function buildFeatures(params) {
    const gpa = safeNum(params.gpa, NaN);
    const calcVal = (params.calcVal || "").trim();

    // Back-compat fields
    const upperCount = Number.isFinite(Number(params.upperCount)) ? parseInt(params.upperCount, 10) : 0;
    const upperRigor = (params.upperRigor || "").trim();

    // New preferred field
    const courses = Array.isArray(params.courses) ? params.courses : null;

    const resumeText = params.resumeText || "";

    // NOTE: index.html currently passes essayMath/essayComm.
    // We keep these names, but the meaning is now:
    // - essayMath = Personal
    // - essayComm = Math + Contribution
    const essayMath = params.essayMath || "";
    const essayComm = params.essayComm || "";

    if (!Number.isFinite(gpa)) throw new Error("Invalid GPA");
    if (!calcVal) throw new Error("Missing calcVal");

    const rt = normText(resumeText);
    const em = normText(essayMath);
    const ec = normText(essayComm);

    const f = [];

    // ---- Academic ----
    const f_gpa = scoreGpa(gpa);       // 0
    const f_calc = scoreCalc(calcVal); // 1

    // Upper-math score:
    // If courses array is provided, use it; otherwise fall back to old fields.
    const f_upper = courses ? scoreUpperFromCourses(courses) : scoreUpperCourses(upperCount, upperRigor || "standard"); // 2

    f.push(f_gpa);
    f.push(f_calc);
    f.push(f_upper);

    // ---- Resume structure + length ----
    f.push(logNormLen(resumeText)); // 3
    f.push(containsAny(rt, ["education", "coursework", "academic"]) ? 10 : 0); // 4
    f.push(containsAny(rt, ["experience", "employment", "intern", "internship"]) ? 10 : 0); // 5
    f.push(containsAny(rt, ["projects", "project"]) ? 10 : 0); // 6
    f.push(containsAny(rt, ["skills", "technical", "languages", "tools"]) ? 10 : 0); // 7

    // ---- Resume keywords (unique-hit saturation) ----
    f.push(clamp(10 * satExp(countHits(rt, KW.math_engagement), 0.35), 0, 10));       // 8
    f.push(clamp(10 * satExp(countHits(rt, KW.research_exposition), 0.35), 0, 10));   // 9
    f.push(clamp(10 * satExp(countHits(rt, KW.teaching_mentoring), 0.55), 0, 10));    // 10
    f.push(clamp(10 * satExp(countHits(rt, KW.leadership_service), 0.40), 0, 10));    // 11
    f.push(clamp(10 * satExp(countHits(rt, KW.awards_honors), 0.40), 0, 10));         // 12
    f.push(clamp(10 * satExp(countHits(rt, KW.competitions), 0.55), 0, 10));          // 13

    // ---- Essays (2–4 sentences) ----
    // Slot 14: band-pass score (personal)
    const f_personal_band = bandpassSentenceScore(essayMath);

    // Slot 15: band-pass score (math+contribution)
    const f_mathcontrib_band = bandpassSentenceScore(essayComm);

    // Slot 16: "personal arc / reflection" (context + action + reflection), blended with light reasoning markers
    const f_personal_arc = clamp(0.70 * personalArcScore(essayMath) + 0.30 * reasoningMarkerScore(essayMath), 0, 10);

    // Slot 17: "plan specificity" for math+contribution, blended with event keywords
    const f_plan = planSpecificityScore(essayComm);

    // Slot 18–19: math terms presence (do not over-reward; saturating)
    const f_personal_math_terms = mathTermScore(essayMath);
    const f_mathcontrib_math_terms = mathTermScore(essayComm);

    f.push(f_personal_band);          // 14
    f.push(f_mathcontrib_band);       // 15
    f.push(f_personal_arc);           // 16
    f.push(f_plan);                   // 17
    f.push(f_personal_math_terms);    // 18
    f.push(f_mathcontrib_math_terms); // 19

    // ---- Penalty (learnable feature) ----
    f.push(densityPenalty(resumeText + "\n" + essayMath + "\n" + essayComm)); // 20

    if (f.length !== FEATURE_NAMES.length) {
      throw new Error(`Feature length mismatch: got ${f.length}, expected ${FEATURE_NAMES.length}`);
    }

    // Return with 6-decimal stability
    return f.map((v) => round6(clamp(v, 0, 10)));
  }

  // ==========
  // Public exports (browser)
  // ==========
  global.PMEFeatures = {
    FEATURE_VERSION,
    FEATURE_NAMES,
    KW,

    // exposed scorers
    scoreGpa,
    scoreCalc,
    scoreUpperCourses,
    scoreUpperFromCourses,

    // exposed essay helpers (useful for debugging)
    bandpassSentenceScore,
    personalArcScore,
    planSpecificityScore,

    // main
    buildFeatures,

    // helpers
    safeNum,
    clamp,
    logNormLen
  };

})(window);