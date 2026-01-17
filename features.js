// features.js
// Shared deterministic feature extraction for PME learning-to-rank (budget $0).
// Used by: tools/feature_extractor.html, index.html (applicant page), and any future tooling.

(function (global) {
  "use strict";

  // ==========
  // Feature schema
  // ==========
  const FEATURE_VERSION = 1;

  // IMPORTANT: This order must remain stable once you begin collecting comparisons.
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

    "essay_math_len_log",
    "essay_comm_len_log",
    "essay_math_reasoning_markers",
    "essay_comm_specificity_markers",
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
    reasoning_markers: ["assume", "suppose", "let", "then", "therefore", "hence", "thus", "it follows"],
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
      "problem set"
    ]
  };

  // ==========
  // Numeric helpers
  // ==========
  function clamp(x, a, b) {
    return Math.max(a, Math.min(b, x));
  }

  function safeNum(x, fallback = NaN) {
    const v = Number(x);
    return Number.isFinite(v) ? v : fallback;
  }

  function round1(x) {
    return Math.round(x * 10) / 10;
  }

  function logNormLen(s) {
    const n = (s || "").trim().length;
    // log(1+n) scaled roughly into [0,10] for typical resume/essay lengths
    return clamp((Math.log1p(n) / Math.log(1 + 6000)) * 10, 0, 10);
  }

  // ==========
  // Text helpers
  // ==========
  function containsAny(text, arr) {
    const t = (text || "").toLowerCase();
    return arr.some((k) => t.includes(k));
  }

  function countHits(text, arr) {
    const t = (text || "").toLowerCase();
    let hits = 0;
    for (const k of arr) {
      if (!k) continue;
      if (t.includes(k)) hits += 1; // unique-hit count (not frequency)
    }
    return hits;
  }

  function densityPenalty(text) {
    const t = (text || "").toLowerCase();
    const n = t.trim().length;
    if (n < 200) return 0;

    const totalHits =
      countHits(t, KW.math_engagement) +
      countHits(t, KW.research_exposition) +
      countHits(t, KW.teaching_mentoring) +
      countHits(t, KW.leadership_service) +
      countHits(t, KW.awards_honors) +
      countHits(t, KW.competitions) +
      countHits(t, KW.math_terms);

    const rate = totalHits / (n / 1000); // hits per 1000 chars
    if (rate <= 40) return 0; // tolerate normal keyword rates

    return clamp(((rate - 40) / 30) * 10, 0, 10);
  }

  // ==========
  // Academic scoring (0–10)
  // ==========
  function scoreGpa(gpa) {
    const x = clamp(gpa, 0, 4);

    // Piecewise mapping:
    // >=3.90 -> 10
    // 3.70–3.89 -> 9–9.9
    // 3.40–3.69 -> 6–8.9
    // 3.20–3.39 -> 3–5.9
    // <3.20 -> 0–2.9
    if (x >= 3.9) return 10;

    if (x >= 3.7) return 9 + (x - 3.7) * (0.9 / 0.19);

    if (x >= 3.4) return 6 + (x - 3.4) * (2.9 / 0.29);

    if (x >= 3.2) return 3 + (x - 3.2) * (2.9 / 0.19);

    // 0.00 => 0, 3.19 => 2.9
    return (x / 3.19) * 2.9;
  }

  function scoreCalc(val) {
    if (val === "yes") return 10;
    if (val === "no") return 0;
    return 0;
  }

  function scoreUpperCourses(count, rigor) {
    const n = clamp(count || 0, 0, 20);

    let base = 0;
    if (n === 0) base = 0;
    else if (n === 1) base = 3.5;
    else if (n === 2) base = 6.0;
    else if (n === 3) base = 7.2;
    else if (n === 4) base = 8.2;
    else if (n === 5) base = 9.0;
    else base = 9.4;

    let bonus = 0;
    if (rigor === "proof") bonus = 0.3;
    if (rigor === "upper") bonus = 0.6;
    if (rigor === "grad") bonus = 0.9;

    return clamp(base + bonus, 0, 10);
  }

  // ==========
  // Main: build feature vector
  // ==========
  function buildFeatures(params) {
    const gpa = safeNum(params.gpa, NaN);
    const calcVal = (params.calcVal || "").trim();
    const upperCount = Number.isFinite(Number(params.upperCount)) ? parseInt(params.upperCount, 10) : 0;
    const upperRigor = (params.upperRigor || "").trim();

    const resumeText = params.resumeText || "";
    const essayMath = params.essayMath || "";
    const essayComm = params.essayComm || "";

    if (!Number.isFinite(gpa)) throw new Error("Invalid GPA");
    if (!calcVal) throw new Error("Missing calcVal");
    if (!upperRigor) throw new Error("Missing upperRigor");

    const rt = resumeText.toLowerCase();
    const em = essayMath.toLowerCase();
    const ec = essayComm.toLowerCase();

    const f = [];

    // Academic
    f.push(scoreGpa(gpa));                              // 0
    f.push(scoreCalc(calcVal));                         // 1
    f.push(scoreUpperCourses(upperCount, upperRigor));  // 2

    // Resume structure + length
    f.push(logNormLen(resumeText)); // 3
    f.push(containsAny(rt, ["education", "coursework", "academic"]) ? 10 : 0); // 4
    f.push(containsAny(rt, ["experience", "employment", "intern", "internship"]) ? 10 : 0); // 5
    f.push(containsAny(rt, ["projects", "project"]) ? 10 : 0); // 6
    f.push(containsAny(rt, ["skills", "technical", "languages", "tools"]) ? 10 : 0); // 7

    // Resume keywords (unique-hit saturation)
    f.push(clamp(countHits(rt, KW.math_engagement) * 1.6, 0, 10));     // 8
    f.push(clamp(countHits(rt, KW.research_exposition) * 1.6, 0, 10)); // 9
    f.push(clamp(countHits(rt, KW.teaching_mentoring) * 2.0, 0, 10));  // 10
    f.push(clamp(countHits(rt, KW.leadership_service) * 1.6, 0, 10));  // 11
    f.push(clamp(countHits(rt, KW.awards_honors) * 1.6, 0, 10));       // 12
    f.push(clamp(countHits(rt, KW.competitions) * 2.2, 0, 10));        // 13

    // Essays
    f.push(logNormLen(essayMath));                                      // 14
    f.push(logNormLen(essayComm));                                      // 15
    f.push(clamp(countHits(em, KW.reasoning_markers) * 2.0, 0, 10));    // 16
    f.push(clamp(countHits(ec, KW.specificity_markers) * 2.0, 0, 10));  // 17
    f.push(clamp(countHits(em, KW.math_terms) * 0.8, 0, 10));           // 18
    f.push(clamp(countHits(ec, KW.math_terms) * 0.8, 0, 10));           // 19

    // Penalty as a learnable feature
    f.push(densityPenalty(resumeText + "\n" + essayMath + "\n" + essayComm)); // 20

    if (f.length !== FEATURE_NAMES.length) {
      throw new Error(`Feature length mismatch: got ${f.length}, expected ${FEATURE_NAMES.length}`);
    }

    // Round to 1 decimal for stability
    return f.map(round1);
  }

  // ==========
  // Public exports (browser)
  // ==========
  global.PMEFeatures = {
    FEATURE_VERSION,
    FEATURE_NAMES,
    KW,
    scoreGpa,
    scoreCalc,
    scoreUpperCourses,
    buildFeatures,
    // helpers:
    safeNum,
    clamp,
    logNormLen
  };

})(window);