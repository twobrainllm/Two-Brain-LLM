/**
 * Two-Brain Chat -- profiler formulas.
 *
 * Pure functions only, ported from the real Python router so the numbers
 * are algorithmically real even though no model/network call backs them
 * yet (see ui/README.md). Every constant below is copied from a captured
 * data/ file, cited inline; every formula mirrors its Python source
 * 1:1 -- if the Python side changes, this module goes stale and should be
 * re-ported, not re-derived from scratch.
 */

// Port of signals/difficulty.py's HARD_QUERY_MARKERS + DifficultyEstimator.score.
const HARD_QUERY_MARKERS = [
  "prove", "derive", "optimi", "algorithm", "complexity", "multi-step",
  "compare and contrast", "write a function", "debug", "root cause",
  "step by step", "architecture", "trade-off", "tradeoff",
];

const ESCALATE_THRESHOLD = 0.55; // routing/policy.py RoutePolicy.escalate_threshold
const LOCAL_LATENCY_BUDGET_MS = 3000; // routing/policy.py RoutePolicy.local_latency_budget_ms

function scoreDifficulty(query) {
  const q = query.toLowerCase();
  let score = 0;
  score += Math.min(query.length / 400.0, 0.4);
  score += 0.15 * HARD_QUERY_MARKERS.filter((m) => q.includes(m)).length;
  score += q.split("?").length - 1 > 1 ? 0.2 : 0;
  return Math.min(score, 1.0);
}

// Port of privacy/patterns.py's PATTERNS + privacy/guard.py's PIIGuard.detect.
const PII_PATTERNS = {
  EMAIL: /\b[\w.+-]+@[\w-]+\.[\w.-]+\b/g,
  PHONE: /\b(?:\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b/g,
  SSN: /\b\d{3}-\d{2}-\d{4}\b/g,
  CREDIT_CARD: /\b(?:\d[ -]*?){13,16}\b/g,
};

function detectPII(text) {
  const entities = [];
  for (const [type, pattern] of Object.entries(PII_PATTERNS)) {
    const matches = text.match(pattern) || [];
    if (matches.length) entities.push({ type, count: matches.length });
  }
  return entities;
}

// Real captured profile constants -- data/profile_workload/pc_3b.json and
// cloud_large.json. Not live measurements; copied values, cited here so a
// change to those files is a deliberate re-port, not silent drift.
const PROFILE = {
  local: {
    source: "data/profile_workload/pc_3b.json",
    // Re-ported after the Shape B re-profile (n=8 through NpuFastBrain itself).
    // Previously 142 / 94.5, measured on a different prompt and a runaway
    // generation that drifted into larger context buckets.
    ttftMeanMs: 105.9,
    perTokenMeanMs: 74.2,
    tokenCostUsdPer1k: 0.0,
  },
  cloud: {
    source: "data/profile_workload/cloud_large.json",
    // Real Cirrascale AI Suite measurement (Llama-3.3-70B), replacing the old
    // datasheet guess -- previously 45 / 320 / 12.4 / 1.8, the last of which
    // made a single query look like $1.03. ttft isn't separable without
    // streaming, so it's folded into networkRttMeanMs (0 here), matching
    // cloud_large.json's own _ttft_note.
    networkRttMeanMs: 263,
    ttftMeanMs: 0,
    perTokenMeanMs: 19.0,
    tokenCostUsdPer1k: 0.00069,
  },
};

// Port of routing/policy.py RoutePolicy.estimate_local_latency_ms's token estimate.
function estimateTokens(query) {
  return Math.max(query.trim().split(/\s+/).filter(Boolean).length * 2, 16);
}

// Port of routing/policy.py's estimate_local_latency_ms (local) and
// routing/brains.py's CloudDeepBrain.answer latency formula (cloud, +RTT).
function estimateLatencyMs(tier, query) {
  const n = estimateTokens(query);
  if (tier === "cloud") {
    const p = PROFILE.cloud;
    return p.networkRttMeanMs + p.ttftMeanMs + n * p.perTokenMeanMs;
  }
  const p = PROFILE.local;
  return p.ttftMeanMs + n * p.perTokenMeanMs;
}

// Port of routing/brains.py CloudDeepBrain.answer's cost formula.
function estimateCostUsd(tier, query) {
  const n = estimateTokens(query);
  const p = tier === "cloud" ? PROFILE.cloud : PROFILE.local;
  return (n / 1000.0) * p.tokenCostUsdPer1k;
}

// Port of routing/policy.py's should_escalate -- always checked against the
// LOCAL tier's own latency estimate, exactly like router.py does, regardless
// of which tier is actually about to answer.
function wouldEscalate(difficulty, localLatencyEstMs) {
  return difficulty >= ESCALATE_THRESHOLD || localLatencyEstMs > LOCAL_LATENCY_BUDGET_MS;
}

// Port of routing/policy.py's escalation_note/local_note text templates.
// `tier` is a manual mock override, not a real decision, so it can disagree
// with what should_escalate would actually pick (e.g. a hard query forced
// to "local"). Rather than print a note whose own numbers contradict its
// claim, say so plainly when that happens instead of picking a template by
// tier alone.
function routingNote(tier, escalate, difficulty, localLatencyEstMs) {
  const real = escalate
    ? `escalating: difficulty=${difficulty.toFixed(2)} (threshold ${ESCALATE_THRESHOLD}) ` +
      `or local_latency_est=${localLatencyEstMs.toFixed(0)}ms > budget ${LOCAL_LATENCY_BUDGET_MS}ms`
    : `answering locally: difficulty=${difficulty.toFixed(2)} < threshold ${ESCALATE_THRESHOLD}, ` +
      `local_latency_est=${localLatencyEstMs.toFixed(0)}ms within budget`;

  if ((tier === "cloud") === escalate) return real;
  return `manually forced to ${tier} — the router would normally ${escalate ? "escalate" : "answer locally"} here (${real})`;
}

// Port of routing/router.py's masking note (router.py:70-75).
function maskingNote(piiEntities) {
  const n = piiEntities.reduce((sum, e) => sum + e.count, 0);
  if (!n) return null;
  return `masked ${n} PII entit${n === 1 ? "y" : "ies"} before any routing decision`;
}

// data/hardware_detect/ai_pc.json -- real quad-client detect capture.
const DEVICE_CONTEXT = "AI PC · Snapdragon® X Elite X1E80100 · Hexagon NPU v73 @ 45 TOPS · 12 cores";

/**
 * Computes the full metrics bundle for one query, mirroring RouteDecision's
 * shape (routing/policy.py). `tier` is whatever the sidebar mock toggle
 * picked -- this does not decide routing, it only profiles the query as if
 * that tier had answered it.
 */
function computeMetrics(query, tier) {
  const difficulty = scoreDifficulty(query);
  const piiEntities = detectPII(query);
  // should_escalate always checks the local estimate, not whichever tier
  // the mock toggle happens to be set to (routing/router.py's own order).
  const localLatencyEstMs = estimateLatencyMs("local", query);
  const escalate = wouldEscalate(difficulty, localLatencyEstMs);
  const estLatencyMs = estimateLatencyMs(tier, query);

  const notes = [routingNote(tier, escalate, difficulty, localLatencyEstMs)];
  const maskNote = maskingNote(piiEntities);
  if (maskNote) notes.unshift(maskNote);

  return {
    difficulty,
    escalateThreshold: ESCALATE_THRESHOLD,
    wouldEscalate: escalate,
    piiEntities,
    piiCount: piiEntities.reduce((sum, e) => sum + e.count, 0),
    estLatencyMs,
    estCostUsd: estimateCostUsd(tier, query),
    notes,
    deviceContext: DEVICE_CONTEXT,
  };
}
