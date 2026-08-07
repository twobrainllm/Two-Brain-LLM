/**
 * Two-Brain Chat -- UI shell.
 *
 * Wired to the real router: `sendToRouter()` POSTs to
 * `src/two_brain_router/api.py`'s `/route` endpoint (see ui/README.md for
 * how to start it). `mockRespond()` / `profiler.js`'s `computeMetrics()`
 * are kept, not deleted -- they're now the **offline fallback**: if the API
 * is unreachable (server not started, wrong port), a reply still renders,
 * labeled "(offline preview)" rather than failing silently or throwing an
 * error at the user. Every message remembers which path answered it
 * (`msg.live`), so switching backend availability later never repaints
 * history -- same principle this file already used for `tier`/`metrics`.
 */

const STORAGE_KEY = "twoBrainChats";
const GROUP_ORDER = ["Today", "Yesterday", "Previous 7 Days", "Previous 30 Days", "Older"];

//: `src/two_brain_router/api.py`'s default bind address/port.
/**
 * `src/two_brain_router/api.py`'s port, on whatever host is serving this page.
 *
 * Derived rather than hardcoded so the UI works from a phone: opened over the
 * LAN, `127.0.0.1` would mean *the phone itself*, which is not running the
 * router. Falls back to loopback for `file://`, where there is no host to
 * borrow. Override with `?api=http://host:port` when the API is somewhere
 * else entirely.
 *
 * Reaching this from another device also needs api.py started with
 * `--host 0.0.0.0`; loopback-only is its deliberate default.
 */
const API_BASE_URL = (() => {
  const override = new URLSearchParams(window.location.search).get("api");
  if (override) return override.replace(/\/$/, "");

  const { protocol, hostname } = window.location;
  if (protocol === "file:" || !hostname) return "http://127.0.0.1:8765";

  // Served over HTTPS means ui/serve_https.py, which proxies /route and
  // /health on its own origin. Use that origin: an HTTPS page is forbidden
  // from fetching an http:// URL (mixed content), so reaching out to
  // :8765 directly would be blocked by the browser regardless of ports.
  if (protocol === "https:") return "";

  return `${protocol}//${hostname}:8765`;
})();

// data/hardware_detect/{ai_pc,mobile}.json -- real quad-client detect /
// adb shell captures, matching profiler.js's own DEVICE_CONTEXT convention.
const DEVICE_CONTEXT_BY_TIER = {
  pc: "AI PC · Snapdragon® X Elite X1E80100 · Hexagon NPU v73 @ 45 TOPS · 12 cores",
  mobile: "Mobile · Snapdragon® 8 Elite (SM8750) · 8 cores · ~10.9 GB RAM · Android 16",
};

const els = {
  chatList: document.getElementById("chat-list"),
  searchInput: document.getElementById("search-input"),
  newChatBtn: document.getElementById("new-chat-btn"),
  emptyState: document.getElementById("empty-state"),
  emptyStateAvatar: document.getElementById("empty-state-avatar"),
  messages: document.getElementById("messages"),
  composer: document.getElementById("composer"),
  composerInner: document.querySelector(".composer-inner"),
  composerInput: document.getElementById("composer-input"),
  composerAttachments: document.getElementById("composer-attachments"),
  dictation: document.getElementById("dictation"),
  dictationWave: document.getElementById("dictation-wave"),
  dictationTime: document.getElementById("dictation-time"),
  dictationTranscript: document.getElementById("dictation-transcript"),
  dictationCancel: document.getElementById("dictation-cancel"),
  dictationConfirm: document.getElementById("dictation-confirm"),
  videomode: document.getElementById("videomode"),
  videomodePreview: document.getElementById("videomode-preview"),
  videomodeClose: document.getElementById("videomode-close"),
  videomodeFlip: document.getElementById("videomode-flip"),
  videomodeShutter: document.getElementById("videomode-shutter"),
  videomodeStatus: document.getElementById("videomode-status"),
  sendBtn: document.getElementById("send-btn"),
  fileInput: document.getElementById("file-input"),
  imageInput: document.getElementById("image-input"),
  videoInput: document.getElementById("video-input"),
  attachFileBtn: document.getElementById("attach-file-btn"),
  attachImageBtn: document.getElementById("attach-image-btn"),
  attachVideoBtn: document.getElementById("attach-video-btn"),
  videoModeBtn: document.getElementById("video-mode-btn"),
  voiceModeBtn: document.getElementById("voice-mode-btn"),
  brainToggle: document.getElementById("brain-toggle"),
  exprButtons: document.getElementById("expr-buttons"),
  robotTemplate: document.getElementById("robot-svg-template"),
  profiler: document.getElementById("profiler"),
  profilerPill: document.getElementById("profiler-pill"),
  profilerCard: document.getElementById("profiler-card"),
  profilerDot: document.getElementById("profiler-dot"),
  profilerSummary: document.getElementById("profiler-summary"),
  profilerCardDot: document.getElementById("profiler-card-dot"),
  profilerCardTier: document.getElementById("profiler-card-tier"),
  profilerDifficultyValue: document.getElementById("profiler-difficulty-value"),
  profilerGaugeFill: document.getElementById("profiler-gauge-fill"),
  profilerPrivacyValue: document.getElementById("profiler-privacy-value"),
  profilerLatencyValue: document.getElementById("profiler-latency-value"),
  profilerCostValue: document.getElementById("profiler-cost-value"),
  profilerNotes: document.getElementById("profiler-notes"),
  profilerFooter: document.getElementById("profiler-footer"),
  backendStatus: document.getElementById("backend-status"),
};

const LOOK_DIRECTIONS = ["left", "right", "up", "down"];
const LOOK_VARIANTS = ["look", "shrink-look", "expand-look"];
const LOOK_EXPRESSIONS = LOOK_VARIANTS.flatMap((variant) => LOOK_DIRECTIONS.map((dir) => `${variant}-${dir}`));
const EXPRESSIONS = ["happy", "wink", "surprised", ...LOOK_EXPRESSIONS];
const EXPRESSION_HOLD_MS = { happy: 500, wink: 900, surprised: 700 };
for (const name of EXPRESSIONS) {
  if (!(name in EXPRESSION_HOLD_MS)) EXPRESSION_HOLD_MS[name] = 1600;
}

/**
 * "Thinking" choreography for the *offline-fallback* reply delay only --
 * not a real signal, just personality. Cloud gets a longer, more deliberate
 * look-around (bigger brain, harder problem); local gets a quick glance.
 * The live path (routing through /route for real) doesn't use this at all
 * -- see `playThinkingLooksUntilSettled`, which paces off the actual
 * network call instead of a canned guess.
 */
const THINK_MS_BY_TIER = { local: 900, cloud: 2800 };
const THINK_RHYTHM_MS = 480;
const HAPPY_LEAD_MS = 450;
const LONG_QUERY_CHARS = 120;

function shuffled(arr) {
  const copy = [...arr];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Cycles the avatar through look/shrink-look/expand-look in random order,
 * for as many complete rhythmMs-length ticks as fit in budgetMs (at least
 * one). Awaits every tick fully -- whatever runs next (e.g. "happy") only
 * ever starts once the last look has actually finished its beat, instead
 * of cutting it off mid-animation at an arbitrary elapsed-time cutoff.
 */
async function playThinkingLooks(avatarEl, budgetMs, rhythmMs) {
  const order = shuffled(LOOK_EXPRESSIONS);
  const steps = Math.max(Math.floor(budgetMs / rhythmMs), 1);
  for (let i = 0; i < steps; i++) {
    playExpression(avatarEl, order[i % order.length], rhythmMs);
    await sleep(rhythmMs);
  }
}

/**
 * Same loop as `playThinkingLooks`, but for a real network call whose
 * duration isn't known up front: loops until `pending` settles instead of
 * for a fixed budget, so the animation actually reflects how long the
 * router took rather than a canned per-tier guess. At least one full beat
 * always plays, so a very fast reply doesn't skip the "thinking" state
 * entirely.
 */
async function playThinkingLooksUntilSettled(avatarEl, pending, rhythmMs) {
  const order = shuffled(LOOK_EXPRESSIONS);
  let settled = false;
  pending.then(
    () => (settled = true),
    () => (settled = true)
  );
  let i = 0;
  do {
    playExpression(avatarEl, order[i % order.length], rhythmMs);
    i++;
    await sleep(rhythmMs);
  } while (!settled);
}

/**
 * Plays a one-off expression on an avatar, then reverts to its normal
 * idle breathing/blink. `name` must be one of EXPRESSIONS.
 *
 * Cancels any expression animation still in flight before switching --
 * a class swap alone can leave the old animation's current frame (e.g.
 * a wink's closed eye) rendered for a tick before the new one takes
 * over, since the browser doesn't restart a still-running animation
 * until its next style recalc.
 */
function playExpression(avatarEl, name, holdMs) {
  if (!avatarEl) return;
  for (const el of avatarEl.querySelectorAll(".rb-eye, .rb-eye-wrap, .rb-face-inner")) {
    for (const anim of el.getAnimations()) anim.cancel();
  }
  for (const e of EXPRESSIONS) avatarEl.classList.remove(`expr-${e}`);
  clearTimeout(avatarEl._exprTimeout);
  avatarEl.classList.add(`expr-${name}`);
  avatarEl._exprTimeout = setTimeout(() => {
    avatarEl.classList.remove(`expr-${name}`);
  }, holdMs ?? EXPRESSION_HOLD_MS[name] ?? 900);
}

function activePreviewAvatar() {
  if (els.messages.style.display !== "none") {
    const last = els.messages.querySelector(".message.assistant:last-child .robot-avatar");
    if (last) return last;
  }
  return els.emptyStateAvatar;
}

/** Badge text per `tier_answered`. `"hybrid"` is a real third outcome, not a
 * flavour of "cloud": the on-device model answered part of the query and only
 * the part it flagged as beyond it was sent on. Saying "Cloud brain" there
 * would understate what stayed local, and "Local brain" would hide that
 * anything left at all. */
const TIER_BADGE_LABEL = {
  local: "Local brain",
  cloud: "Cloud brain",
  hybrid: "Local brain + cloud (split)",
};

const state = {
  chats: loadChats(),
  activeChatId: null,
  // Only consulted on the offline-fallback path now (see mockRespond /
  // computeMetrics below) -- when the API answers for real, tier comes back
  // in the response (`tier_answered`), it is not chosen up front.
  currentTier: "local",
  searchQuery: "",
  backendLive: null, // null = not checked yet, true/false after checkBackend()
  attachments: [],
  videoMode: false,
  listening: false,
};

function loadChats() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveChats() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state.chats));
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function robotSvgMarkup() {
  return els.robotTemplate.innerHTML;
}

function groupLabelFor(timestamp, now) {
  const startOfToday = new Date(now);
  startOfToday.setHours(0, 0, 0, 0);
  const startOfDay = new Date(timestamp);
  startOfDay.setHours(0, 0, 0, 0);
  const diffDays = Math.round((startOfToday - startOfDay) / 86400000);
  if (diffDays <= 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  if (diffDays <= 7) return "Previous 7 Days";
  if (diffDays <= 30) return "Previous 30 Days";
  return "Older";
}

function chatMatchesSearch(chat, query) {
  if (!query) return true;
  const q = query.toLowerCase();
  if (chat.title.toLowerCase().includes(q)) return true;
  return chat.messages.some(
    (m) =>
      m.content.toLowerCase().includes(q) ||
      m.attachments?.some((a) => a.name.toLowerCase().includes(q))
  );
}

function renderChatList() {
  const now = Date.now();
  const visible = state.chats
    .filter((c) => chatMatchesSearch(c, state.searchQuery))
    .sort((a, b) => b.updatedAt - a.updatedAt);

  els.chatList.innerHTML = "";

  if (visible.length === 0) {
    const empty = document.createElement("div");
    empty.className = "chat-list-empty";
    empty.textContent = state.searchQuery ? "No chats match your search." : "No chats yet.";
    els.chatList.appendChild(empty);
    return;
  }

  const grouped = new Map();
  for (const chat of visible) {
    const label = groupLabelFor(chat.updatedAt, now);
    if (!grouped.has(label)) grouped.set(label, []);
    grouped.get(label).push(chat);
  }

  for (const label of GROUP_ORDER) {
    const chatsInGroup = grouped.get(label);
    if (!chatsInGroup) continue;

    const header = document.createElement("div");
    header.className = "chat-group-label";
    header.textContent = label;
    els.chatList.appendChild(header);

    for (const chat of chatsInGroup) {
      els.chatList.appendChild(renderChatItem(chat));
    }
  }
}

function renderChatItem(chat) {
  const item = document.createElement("div");
  item.className = "chat-item" + (chat.id === state.activeChatId ? " active" : "");
  item.dataset.chatId = chat.id;

  const title = document.createElement("span");
  title.className = "chat-item-title";
  title.textContent = chat.title;
  item.appendChild(title);

  const del = document.createElement("button");
  del.className = "chat-item-delete";
  del.type = "button";
  del.setAttribute("aria-label", `Delete "${chat.title}"`);
  del.innerHTML = '<svg viewBox="0 0 20 20" width="14" height="14" fill="none"><path d="M4 6h12M8 6V4.5A1 1 0 0 1 9 3.5h2a1 1 0 0 1 1 1V6M6 6l.6 10a1 1 0 0 0 1 1h4.8a1 1 0 0 0 1-1L14 6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  del.addEventListener("click", (e) => {
    e.stopPropagation();
    deleteChat(chat.id);
  });
  item.appendChild(del);

  item.addEventListener("click", () => openChat(chat.id));
  return item;
}

function deleteChat(chatId) {
  const chat = state.chats.find((c) => c.id === chatId);
  if (!chat) return;
  if (!confirm(`Delete "${chat.title}"? This can't be undone.`)) return;

  state.chats = state.chats.filter((c) => c.id !== chatId);
  saveChats();

  if (state.activeChatId === chatId) {
    state.activeChatId = null;
    showEmptyState();
  }
  renderChatList();
}

function openChat(chatId) {
  const chat = state.chats.find((c) => c.id === chatId);
  if (!chat) return;
  state.activeChatId = chatId;
  renderMessages(chat);
  renderChatList();

  const lastAssistant = [...chat.messages].reverse().find((m) => m.role === "assistant" && m.metrics);
  if (lastAssistant) renderProfiler(lastAssistant.metrics, lastAssistant.tier);
  else showProfilerIdle();
}

function showEmptyState() {
  els.emptyState.style.display = "flex";
  els.messages.style.display = "none";
  els.messages.innerHTML = "";
  showProfilerIdle();
}

function renderMessages(chat) {
  els.emptyState.style.display = "none";
  els.messages.style.display = "flex";
  els.messages.innerHTML = "";
  for (const msg of chat.messages) {
    els.messages.appendChild(renderMessageEl(msg));
  }
  scrollToBottom();
}

function renderMessageEl(msg) {
  const row = document.createElement("div");
  row.className = `message ${msg.role}`;

  if (msg.role === "assistant") {
    const avatar = document.createElement("div");
    avatar.className = "robot-avatar";
    avatar.dataset.tier = msg.tier || "local";
    avatar.innerHTML = robotSvgMarkup();
    row.appendChild(avatar);
  }

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  if (msg.role === "assistant") {
    const badge = document.createElement("div");
    badge.className = "tier-badge";
    const tier = msg.tier || "local";
    // `msg.live` is undefined for the "…" placeholder row (no verdict yet)
    // and for any chat saved before this field existed -- treat both as
    // "don't claim either way" rather than defaulting to "(simulated)",
    // which would mislabel old real-seeming history that predates this field.
    const suffix = msg.live === true ? "" : msg.live === false ? " (offline preview)" : "";
    badge.innerHTML =
      `<span class="tier-dot" data-tier="${tier}"></span>` +
      (TIER_BADGE_LABEL[tier] ?? TIER_BADGE_LABEL.local) +
      suffix;
    bubble.appendChild(badge);
  }

  if (msg.attachments?.length) {
    const attachRow = document.createElement("div");
    for (const att of msg.attachments) {
      attachRow.appendChild(attachmentChipEl(att, { removable: false }));
    }
    bubble.appendChild(attachRow);
  }

  if (msg.content) {
    const text = document.createElement("div");
    text.innerHTML = escapeHtml(msg.content).replace(/\n/g, "<br>");
    bubble.appendChild(text);
  }

  row.appendChild(bubble);
  return row;
}

function scrollToBottom() {
  const area = document.getElementById("chat-area");
  area.scrollTop = area.scrollHeight;
}

function titleFromQuery(query) {
  const trimmed = query.trim().replace(/\s+/g, " ");
  return trimmed.length > 48 ? trimmed.slice(0, 48) + "…" : trimmed || "New chat";
}

function createChat(firstMessage) {
  const now = Date.now();
  const chat = {
    id: `chat_${now}_${Math.random().toString(36).slice(2, 8)}`,
    title: titleFromQuery(firstMessage),
    createdAt: now,
    updatedAt: now,
    messages: [],
  };
  state.chats.push(chat);
  state.activeChatId = chat.id;
  return chat;
}

function tierLabel(tier) {
  if (tier === "hybrid") return "Local + Cloud";
  return tier === "cloud" ? "Cloud" : "Local";
}

/** Same thing with the noun attached. Separate from `tierLabel` because
 * "Local + Cloud brain" reads as one brain with a long name -- on a split there
 * were two of them, and the card is where that should be plainest. */
function tierCardLabel(tier) {
  return tier === "hybrid" ? "Local brain + cloud" : `${tierLabel(tier)} brain`;
}

function formatPrivacy(metrics) {
  if (!metrics.piiCount) return "No PII detected";
  const parts = metrics.piiEntities.map((e) => `${e.count} ${e.type}`);
  const found = `${metrics.piiCount} detected (${parts.join(", ")})`;
  // Detected and masked are different numbers now that masking happens at the
  // cloud boundary rather than at the front door. Saying only "N masked" would
  // print "0 masked" for a PII-heavy query answered entirely on-device -- which
  // reads as "we did nothing" when it actually means "none of it left".
  const masked = metrics.piiMasked ?? (metrics.wouldEscalate ? metrics.piiCount : 0);
  return masked === 0
    ? `${found} — none left the device`
    : `${found}, ${masked} masked before leaving`;
}

/** Renders the profiler pill + card from one message's computed metrics (see profiler.js). */
function renderProfiler(metrics, tier) {
  els.profiler.dataset.hasData = "true";
  els.profilerDot.dataset.tier = tier;
  els.profilerCardDot.dataset.tier = tier;
  els.profilerCardTier.textContent = tierCardLabel(tier);

  const summaryLatencyMs = Math.round(metrics.actualLatencyMs ?? metrics.estLatencyMs);
  els.profilerSummary.textContent = `${tierLabel(tier)} · ${summaryLatencyMs}ms`;

  els.profilerDifficultyValue.textContent = `${metrics.difficulty.toFixed(2)} / ${metrics.escalateThreshold}`;
  els.profilerGaugeFill.style.width = `${Math.round(metrics.difficulty * 100)}%`;
  els.profilerGaugeFill.style.background =
    metrics.difficulty >= metrics.escalateThreshold ? "var(--brain-cloud)" : "var(--brain-local)";

  els.profilerPrivacyValue.textContent = formatPrivacy(metrics);

  els.profilerLatencyValue.textContent =
    `est ${Math.round(metrics.estLatencyMs)}ms` +
    (metrics.actualLatencyMs != null ? ` · sim ${Math.round(metrics.actualLatencyMs)}ms` : "");

  els.profilerCostValue.textContent =
    metrics.estCostUsd > 0 ? `$${metrics.estCostUsd.toFixed(5)}` : "$0 (on-device)";

  els.profilerNotes.innerHTML = "";
  for (const note of metrics.notes) {
    const li = document.createElement("li");
    li.textContent = note;
    els.profilerNotes.appendChild(li);
  }

  els.profilerFooter.textContent = metrics.deviceContext;
}

function showProfilerIdle() {
  els.profiler.dataset.hasData = "false";
  els.profilerDot.dataset.tier = "";
  els.profilerSummary.textContent = "Profiler";
  closeProfilerCard();
}

function openProfilerCard() {
  if (els.profiler.dataset.hasData !== "true") return;
  els.profilerCard.dataset.open = "true";
  els.profilerPill.setAttribute("aria-expanded", "true");
}

function closeProfilerCard() {
  els.profilerCard.dataset.open = "false";
  els.profilerPill.setAttribute("aria-expanded", "false");
}

function toggleProfilerCard() {
  if (els.profilerCard.dataset.open === "true") closeProfilerCard();
  else openProfilerCard();
}

/**
 * Real call: POSTs to `two_brain_router.api`'s `/route`. Throws (network
 * error, non-2xx, bad JSON) rather than returning a sentinel -- callers
 * decide what "the API is unreachable" means for them; here that means
 * `handleSend` falls back to `mockRespond`/`computeMetrics`.
 */
async function sendToRouter(query, context) {
  const res = await fetch(`${API_BASE_URL}/route`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, context }),
  });
  const body = await res.json();
  if (!res.ok) throw new Error(body.error || `HTTP ${res.status}`);
  return body;
}

/** Adapts a /route response into the metrics shape `renderProfiler` and
 * `formatPrivacy` already consume (same field names `computeMetrics` in
 * profiler.js produces) -- so neither of those needs to know or care
 * whether the numbers came from a real call or the offline fallback. */
function metricsFromRouteResponse(body) {
  return {
    difficulty: body.difficulty_score,
    escalateThreshold: body.escalate_threshold,
    // Anything that isn't purely local crossed the boundary. `"hybrid"` did
    // reach the cloud -- it just kept the local half too -- so testing for
    // `=== "cloud"` here would have reported a real escalation as none.
    wouldEscalate: body.tier_answered !== "local",
    piiEntities: body.pii_entities,
    // `piiCount` is what the query *contained*; `piiMasked` is what actually
    // had to be masked because it crossed. On a locally-answered query the
    // second is 0 while the first is not, and that gap is the demo.
    piiCount: body.pii_entities_detected,
    piiMasked: body.pii_entities_masked,
    estLatencyMs: body.est_latency_ms,
    estCostUsd: body.est_cost_usd,
    notes: body.notes,
    deviceContext: DEVICE_CONTEXT_BY_TIER[body.tier] ?? body.tier,
  };
}

/** Pings /health once (on load, and again after any failed /route call) so
 * the sidebar can say plainly whether replies are live or falling back --
 * rather than the user discovering it only from a subtle badge change. */
async function checkBackend() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, { method: "GET" });
    const body = await res.json();
    setBackendStatus(res.ok, body.tier);
  } catch {
    setBackendStatus(false);
  }
}

function setBackendStatus(live, tier) {
  state.backendLive = live;
  if (!els.backendStatus) return;
  els.backendStatus.textContent = live
    ? `● Live — routing as ${tier}`
    : "○ API offline — replies use the offline preview";
  els.backendStatus.dataset.live = String(live);
}

/**
 * OFFLINE FALLBACK: stands in for TwoBrainRouter.route(query, context) when
 * `sendToRouter` fails. `tier` is whatever the sidebar's preview toggle is
 * set to -- a deliberate preview choice in this path, unlike the live path
 * where tier is always a real decision returned by the server.
 */
function mockRespond(query, tier) {
  const tierLabel = tier === "cloud" ? "Cloud AI 100 (simulated)" : "local fast brain (simulated)";
  return (
    `This is a simulated reply from the ${tierLabel}.\n\n` +
    `Offline preview: the /route API wasn't reachable at ${API_BASE_URL}, so no model ran ` +
    `and no query left this browser tab. Start it with ` +
    `"python -m two_brain_router.api" and resend to get a real answer. ` +
    `Use the "Preview tier" toggle in the sidebar to see the other color while offline.`
  );
}

/* ---------- Composer attachments ----------
 *
 * Only file *metadata* (name, size, kind) is kept -- never the bytes. Chats
 * live in localStorage (~5MB per origin), so stashing a single PDF there
 * would blow the quota and take the whole history down with it. Nothing
 * reads or uploads the file contents yet either; wiring that up belongs with
 * the `/route` endpoint in README.md's step 1, alongside the repo's existing
 * image path (local VLM sees the image, cloud gets a masked description).
 */

const MAX_ATTACHMENTS = 10;

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function addAttachments(fileList, kind) {
  for (const file of fileList) {
    if (state.attachments.length >= MAX_ATTACHMENTS) break;
    state.attachments.push({
      id: `att_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      name: file.name,
      size: file.size,
      kind,
    });
  }
  renderAttachments();
  updateSendState();
}

function removeAttachment(id) {
  state.attachments = state.attachments.filter((a) => a.id !== id);
  renderAttachments();
  updateSendState();
}

function attachmentChipEl(att, { removable }) {
  const chip = document.createElement("span");
  chip.className = "attachment-chip";

  const ICONS = {
    image:
      '<svg viewBox="0 0 20 20" width="13" height="13" fill="none"><rect x="2.5" y="4" width="15" height="12" rx="2.5" stroke="currentColor" stroke-width="1.5"/><path d="M3 13.5l3.6-3.2a1.5 1.5 0 0 1 2 0L13 14" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>',
    video:
      '<svg viewBox="0 0 20 20" width="13" height="13" fill="none"><rect x="2.5" y="5" width="10.5" height="10" rx="2" stroke="currentColor" stroke-width="1.5"/><path d="M13 9l3.4-2.4a.7.7 0 0 1 1.1.6v5.6a.7.7 0 0 1-1.1.6L13 11z" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/></svg>',
    file:
      '<svg viewBox="0 0 20 20" width="13" height="13" fill="none"><path d="M11.5 2.5H6a1.5 1.5 0 0 0-1.5 1.5v12A1.5 1.5 0 0 0 6 17.5h8a1.5 1.5 0 0 0 1.5-1.5V6.5zM11.5 2.5v4h4" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/></svg>',
  };
  const icon = document.createElement("span");
  icon.innerHTML = ICONS[att.kind] || ICONS.file;
  chip.appendChild(icon);

  const name = document.createElement("span");
  name.className = "attachment-chip-name";
  name.textContent = att.name;
  chip.appendChild(name);

  const size = document.createElement("span");
  size.className = "attachment-chip-size";
  size.textContent = formatBytes(att.size);
  chip.appendChild(size);

  if (removable) {
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "attachment-chip-remove";
    remove.setAttribute("aria-label", `Remove ${att.name}`);
    remove.innerHTML = '<svg viewBox="0 0 20 20" width="11" height="11" fill="none"><path d="M5 5l10 10M15 5L5 15" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>';
    remove.addEventListener("click", () => removeAttachment(att.id));
    chip.appendChild(remove);
  }

  return chip;
}

function renderAttachments() {
  els.composerAttachments.innerHTML = "";
  els.composerAttachments.hidden = state.attachments.length === 0;
  for (const att of state.attachments) {
    els.composerAttachments.appendChild(attachmentChipEl(att, { removable: true }));
  }
}

function clearAttachments() {
  state.attachments = [];
  els.fileInput.value = "";
  els.imageInput.value = "";
  renderAttachments();
}

/* ---------- Dictation (voice -> text) ----------
 *
 * ChatGPT's composer dictation, rebuilt: tapping the mic swaps the textarea
 * for a live waveform + elapsed timer with discard/insert buttons, and the
 * transcript lands in the textarea to edit before sending -- it never sends
 * on its own.
 *
 * Two independent browser APIs run at once, on purpose:
 *   - getUserMedia + AnalyserNode drives the waveform off real mic amplitude,
 *     so the bars reflect the actual signal instead of animating on a timer.
 *   - SpeechRecognition produces the transcript.
 * Neither one alone does both jobs: SpeechRecognition exposes no audio levels,
 * and an AnalyserNode can't transcribe.
 *
 * PRIVACY NOTE, and it matters for this repo specifically: Chrome's
 * SpeechRecognition is *not* on-device -- it streams audio to Google's servers
 * for transcription. For a project whose whole thesis is that the local brain
 * keeps data on the device, dictation is therefore a cloud hop that happens
 * before the router ever sees the query, and it bypasses PIIGuard entirely
 * (privacy/guard.py only ever sees the resulting text). Swapping this for a
 * local Whisper endpoint behind README.md step 1's API server is the fix.
 */

const SpeechRecognitionCtor = window.SpeechRecognition || window.webkitSpeechRecognition;

const WAVE_BARS = 48;
const WAVE_TICK_MS = 55;
const WAVE_GAIN = 2.4;

const dictation = {
  recognition: null,
  stream: null,
  audioCtx: null,
  analyser: null,
  sampleBuf: null,
  rafId: null,
  timerId: null,
  startedAt: 0,
  lastTickAt: 0,
  levels: new Array(WAVE_BARS).fill(0),
  finalText: "",
  interimText: "",
};

function buildWaveBars() {
  els.dictationWave.innerHTML = "";
  for (let i = 0; i < WAVE_BARS; i++) {
    const bar = document.createElement("div");
    bar.className = "wave-bar";
    els.dictationWave.appendChild(bar);
  }
}

function paintWave() {
  const bars = els.dictationWave.children;
  for (let i = 0; i < bars.length; i++) {
    // 2px floor keeps a visible idle line during silence, like ChatGPT's.
    bars[i].style.height = `${2 + dictation.levels[i] * 30}px`;
  }
}

/** RMS of the current frame, 0..1, mildly boosted so speech fills the bar height. */
function currentLevel() {
  dictation.analyser.getByteTimeDomainData(dictation.sampleBuf);
  let sumSquares = 0;
  for (const sample of dictation.sampleBuf) {
    const centered = (sample - 128) / 128;
    sumSquares += centered * centered;
  }
  const rms = Math.sqrt(sumSquares / dictation.sampleBuf.length);
  return Math.min(rms * WAVE_GAIN, 1);
}

/**
 * Mobile's stand-in for the amplitude meter: a travelling wave, so the panel
 * reads as "listening" without pretending to show the microphone signal.
 * See startDictation for why the real meter cannot run here.
 */
function pulseFrame(now) {
  dictation.rafId = requestAnimationFrame(pulseFrame);
  if (now - dictation.lastTickAt < WAVE_TICK_MS) return;
  dictation.lastTickAt = now;
  const t = now / 260;
  for (let i = 0; i < WAVE_BARS; i++) {
    dictation.levels[i] = 0.18 + 0.32 * (Math.sin(t - i / 5) + 1) / 2;
  }
  paintWave();
}

function waveFrame(now) {
  dictation.rafId = requestAnimationFrame(waveFrame);
  if (now - dictation.lastTickAt < WAVE_TICK_MS) return;
  dictation.lastTickAt = now;
  dictation.levels.shift();
  dictation.levels.push(currentLevel());
  paintWave();
}

function formatElapsed(ms) {
  const total = Math.floor(ms / 1000);
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

function renderTranscript() {
  els.dictationTranscript.innerHTML = "";
  if (dictation.finalText) {
    els.dictationTranscript.appendChild(document.createTextNode(dictation.finalText));
  }
  if (dictation.interimText) {
    const interim = document.createElement("span");
    interim.className = "interim";
    interim.textContent = (dictation.finalText ? " " : "") + dictation.interimText;
    els.dictationTranscript.appendChild(interim);
  }
  els.dictationTranscript.scrollTop = els.dictationTranscript.scrollHeight;
  // Never disabled. It was, when there was no transcript yet -- which made the
  // button unpressable in exactly the case where the user most needs a way
  // out, and read as "the tick is broken" rather than "nothing was heard".
  // Confirming with an empty transcript simply closes the panel.
  els.dictationConfirm.disabled = false;
}

function showDictationError(message) {
  els.dictationTranscript.innerHTML = "";
  const err = document.createElement("span");
  err.className = "dictation-error";
  err.textContent = message;
  els.dictationTranscript.appendChild(err);
}

async function startDictation() {
  if (!SpeechRecognitionCtor) return;

  els.composerInner.dataset.dictating = "true";
  els.dictation.hidden = false;
  els.dictation.dataset.tier = state.currentTier;
  els.voiceModeBtn.setAttribute("aria-pressed", "true");
  state.listening = true;

  dictation.finalText = "";
  dictation.interimText = "";
  dictation.levels = new Array(WAVE_BARS).fill(0);
  dictation.startedAt = performance.now();
  buildWaveBars();
  renderTranscript();
  paintWave();

  els.dictationTime.textContent = "0:00";
  dictation.timerId = setInterval(() => {
    els.dictationTime.textContent = formatElapsed(performance.now() - dictation.startedAt);
  }, 200);

  // The waveform is amplitude-driven on desktop, but NOT on mobile, and that
  // is deliberate rather than a shortcut.
  //
  // On Android, holding a getUserMedia stream open starves SpeechRecognition
  // of the microphone: the bars animate beautifully and the transcript stays
  // empty forever. Two APIs, one mic, and recognition loses. Since the
  // transcript is the entire point of dictation and the waveform is only
  // feedback, the waveform is what gives way -- mobile gets an activity
  // animation that is explicitly not claiming to show your voice.
  if (isMobileDevice()) {
    dictation.rafId = requestAnimationFrame(pulseFrame);
  } else {
    try {
      dictation.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      dictation.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      dictation.analyser = dictation.audioCtx.createAnalyser();
      dictation.analyser.fftSize = 1024;
      dictation.sampleBuf = new Uint8Array(dictation.analyser.fftSize);
      dictation.audioCtx.createMediaStreamSource(dictation.stream).connect(dictation.analyser);
      dictation.rafId = requestAnimationFrame(waveFrame);
    } catch {
      /* No mic for the meter; SpeechRecognition below may still be granted. */
    }
  }

  dictation.recognition = new SpeechRecognitionCtor();
  dictation.recognition.continuous = true;
  dictation.recognition.interimResults = true;
  dictation.recognition.lang = navigator.language || "en-US";

  dictation.recognition.addEventListener("result", (e) => {
    let interim = "";
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const chunk = e.results[i][0].transcript;
      if (e.results[i].isFinal) {
        dictation.finalText = (dictation.finalText + " " + chunk.trim()).trim();
      } else {
        interim += chunk;
      }
    }
    dictation.interimText = interim.trim();
    renderTranscript();
  });

  dictation.recognition.addEventListener("error", (e) => {
    if (e.error === "no-speech" || e.error === "aborted") return;
    showDictationError(
      e.error === "not-allowed" || e.error === "service-not-allowed"
        ? "Microphone blocked. Allow mic access for this page, then try again."
        : `Dictation error: ${e.error}`
    );
  });

  // continuous mode still ends itself after a long silence; restart so the
  // session lasts until the user explicitly discards or inserts.
  dictation.recognition.addEventListener("end", () => {
    if (!state.listening) return;
    try {
      dictation.recognition.start();
    } catch {
      /* Already restarting. */
    }
  });

  try {
    dictation.recognition.start();
  } catch {
    /* start() throws if a previous session is still tearing down. */
  }
}

/** Tears down mic, waveform and recognition. Returns the transcript so far. */
function stopDictation() {
  state.listening = false;

  if (dictation.recognition) {
    dictation.recognition.abort();
    dictation.recognition = null;
  }
  if (dictation.rafId) cancelAnimationFrame(dictation.rafId);
  dictation.rafId = null;
  clearInterval(dictation.timerId);
  dictation.timerId = null;

  // Release the mic, or the browser keeps showing a "recording" indicator.
  if (dictation.stream) {
    for (const track of dictation.stream.getTracks()) track.stop();
    dictation.stream = null;
  }
  if (dictation.audioCtx) {
    dictation.audioCtx.close();
    dictation.audioCtx = null;
  }

  els.composerInner.dataset.dictating = "false";
  els.dictation.hidden = true;
  els.voiceModeBtn.setAttribute("aria-pressed", "false");

  return [dictation.finalText, dictation.interimText].filter(Boolean).join(" ").trim();
}

/** Discard: teardown, nothing reaches the composer. */
function cancelDictation() {
  if (!state.listening) return;
  stopDictation();
  els.composerInput.focus();
}

/** Insert: append the transcript to whatever is already typed, for editing. */
function confirmDictation() {
  if (!state.listening) return;
  const transcript = stopDictation();
  if (transcript) {
    const existing = els.composerInput.value.trim();
    els.composerInput.value = existing ? `${existing} ${transcript}` : transcript;
  }
  els.composerInput.focus();
  autoGrow();
  updateSendState();
}

function initVoice() {
  if (!SpeechRecognitionCtor) {
    els.voiceModeBtn.disabled = true;
    els.voiceModeBtn.title = "Dictation unavailable — this browser has no SpeechRecognition API.";
    return;
  }
  if (!window.isSecureContext) {
    els.voiceModeBtn.disabled = true;
    els.voiceModeBtn.title = "Dictation needs a secure context — open this page over https or localhost.";
  }
}

function toggleVoice() {
  if (state.listening) confirmDictation();
  else startDictation();
}

/* ---------- Video mode (camera capture) ----------
 *
 * Real: opens the device camera with getUserMedia, shows a live preview, and
 * captures the current frame to a JPEG attachment. Not a mock.
 *
 * Deliberately mobile-only. The tier this feeds is the Mobile 1B fast brain,
 * and pointing a phone's rear camera at something is the actual interaction
 * being demoed; a laptop webcam pointed at the user's face is a different
 * feature. On desktop the button disables itself and says why, rather than
 * silently doing nothing.
 *
 * Frames are captured in-memory and, like every other attachment here, only
 * their metadata is persisted (see addAttachments) -- a base64 JPEG in
 * localStorage would blow the quota. Sending the pixels anywhere needs
 * README.md step 1's API server; the repo's image path already masks on the
 * far side (local VLM sees the image, cloud gets a masked description).
 */

const CAPTURE_MIME = "image/jpeg";
const CAPTURE_QUALITY = 0.9;

const camera = { stream: null, facing: "environment" };

/**
 * Mobile detection, best signal first. userAgentData.mobile is the only
 * non-heuristic answer but is Chromium-only; the fallback needs both a coarse
 * pointer and real touch points, since either alone matches touchscreen
 * laptops. iPadOS reports itself as a Mac, so maxTouchPoints catches it.
 */
function isMobileDevice() {
  if (typeof navigator.userAgentData?.mobile === "boolean") return navigator.userAgentData.mobile;
  if (/Android|iPhone|iPod|Mobile/i.test(navigator.userAgent)) return true;
  const coarse = window.matchMedia?.("(pointer: coarse)").matches ?? false;
  return coarse && navigator.maxTouchPoints > 1;
}

function setCameraStatus(message) {
  els.videomodeStatus.textContent = message;
}

async function openCameraStream() {
  if (camera.stream) {
    for (const track of camera.stream.getTracks()) track.stop();
    camera.stream = null;
  }
  camera.stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: camera.facing },
    audio: false,
  });
  els.videomodePreview.srcObject = camera.stream;
  els.videomodePreview.dataset.facing = camera.facing;
  await els.videomodePreview.play().catch(() => {});
}

async function startVideoMode() {
  state.videoMode = true;
  els.composerInner.dataset.videomode = "true";
  els.videomode.hidden = false;
  els.videoModeBtn.setAttribute("aria-pressed", "true");
  els.videoModeBtn.dataset.tier = state.currentTier;
  setCameraStatus("");
  els.videomodeShutter.disabled = true;

  try {
    await openCameraStream();
    els.videomodeShutter.disabled = false;
  } catch (err) {
    els.videomodeShutter.disabled = true;
    setCameraStatus(
      err?.name === "NotAllowedError"
        ? "Camera blocked — allow camera access for this page."
        : err?.name === "NotFoundError"
        ? "No camera found on this device."
        : `Camera unavailable: ${err?.name || "unknown error"}`
    );
  }
}

function stopVideoMode() {
  state.videoMode = false;
  if (camera.stream) {
    for (const track of camera.stream.getTracks()) track.stop();
    camera.stream = null;
  }
  els.videomodePreview.srcObject = null;
  els.composerInner.dataset.videomode = "false";
  els.videomode.hidden = true;
  els.videoModeBtn.setAttribute("aria-pressed", "false");
}

async function flipCamera() {
  camera.facing = camera.facing === "environment" ? "user" : "environment";
  try {
    await openCameraStream();
  } catch {
    setCameraStatus("Couldn't switch camera — this device may only have one.");
  }
}

/** Grabs the current preview frame at the stream's native resolution. */
function captureFrame() {
  const video = els.videomodePreview;
  if (!video.videoWidth) return;

  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");

  // Undo the preview's front-camera mirroring so the saved frame matches
  // what the lens actually saw, not the mirror the user was looking at.
  if (camera.facing === "user") {
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(
    (blob) => {
      if (!blob) return;
      const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
      addAttachments([new File([blob], `capture-${stamp}.jpg`, { type: CAPTURE_MIME })], "image");
      stopVideoMode();
      els.composerInput.focus();
    },
    CAPTURE_MIME,
    CAPTURE_QUALITY
  );
}

function toggleVideoMode() {
  if (state.videoMode) stopVideoMode();
  else startVideoMode();
}

function initVideoMode() {
  // Enabled on every device now, not phone-only: a laptop webcam is a
  // legitimate input for the AI PC tier's VLM, and gating it off meant the
  // button looked broken on the machine most of the demo runs on.
  if (!navigator.mediaDevices?.getUserMedia) {
    els.videoModeBtn.disabled = true;
    els.videoModeBtn.title = "This browser has no camera API.";
    return;
  }
  if (!window.isSecureContext) {
    els.videoModeBtn.disabled = true;
    els.videoModeBtn.title =
      "Camera needs a secure context. Serve this page over https " +
      "(ui/serve_https.py) or open it on localhost.";
  }
}

async function handleSend(e) {
  e.preventDefault();
  const query = els.composerInput.value.trim();
  const attachments = state.attachments;
  if (!query && attachments.length === 0) return;

  let chat = state.chats.find((c) => c.id === state.activeChatId);
  if (!chat) chat = createChat(query || attachments[0].name);

  chat.messages.push({ role: "user", content: query, attachments, timestamp: Date.now() });
  chat.updatedAt = Date.now();
  saveChats();
  renderMessages(chat);
  renderChatList();

  els.composerInput.value = "";
  clearAttachments();
  autoGrow();
  updateSendState();

  // Placeholder tier for the thinking avatar only -- real tier isn't known
  // until the response comes back (or the fallback kicks in below).
  const thinkingRow = renderMessageEl({ role: "assistant", content: "…", tier: "local" });
  const thinkingAvatar = thinkingRow.querySelector(".robot-avatar");
  thinkingAvatar?.classList.add("thinking");
  els.messages.appendChild(thinkingRow);
  scrollToBottom();
  const thinkingStartedAt = performance.now();

  const isLongQuery = query.length > LONG_QUERY_CHARS;
  if (isLongQuery) {
    playExpression(thinkingAvatar, "surprised");
    await sleep(EXPRESSION_HOLD_MS.surprised);
  }

  // In flight immediately; the animation below just watches it settle.
  const routed = sendToRouter(query, "");
  const looping = playThinkingLooksUntilSettled(thinkingAvatar, routed, THINK_RHYTHM_MS);

  let tier, answer, metrics, live;
  try {
    const body = await routed;
    await looping; // let the current beat finish instead of cutting it off
    tier = body.tier_answered;
    answer = body.answer;
    metrics = metricsFromRouteResponse(body);
    live = true;
    setBackendStatus(true, body.tier);
  } catch (err) {
    console.warn("two-brain-router API unreachable, using offline preview:", err);
    // `routed` rejects fast (a refused connection, not a real wait), so
    // `looping` already stopped -- swap to the personality-paced budget
    // loop instead of a near-instant reply, same rhythm the mock always had.
    const fallbackTier = state.currentTier;
    await playThinkingLooks(
      thinkingAvatar,
      THINK_MS_BY_TIER[fallbackTier] ?? THINK_MS_BY_TIER.local,
      THINK_RHYTHM_MS
    );
    tier = fallbackTier;
    answer = mockRespond(query, tier);
    metrics = computeMetrics(query, tier);
    live = false;
    setBackendStatus(false);
  }

  playExpression(thinkingAvatar, "happy", HAPPY_LEAD_MS);
  await sleep(HAPPY_LEAD_MS);

  metrics.actualLatencyMs = performance.now() - thinkingStartedAt;
  chat.messages.push({ role: "assistant", content: answer, tier, live, timestamp: Date.now(), metrics });
  chat.updatedAt = Date.now();
  saveChats();
  renderMessages(chat);
  renderChatList();
  playExpression(els.messages.querySelector(".message.assistant:last-child .robot-avatar"), "happy");
  renderProfiler(metrics, tier);
}

function autoGrow() {
  els.composerInput.style.height = "auto";
  els.composerInput.style.height = Math.min(els.composerInput.scrollHeight, 200) + "px";
}

function updateSendState() {
  const hasText = els.composerInput.value.trim().length > 0;
  els.sendBtn.disabled = !hasText && state.attachments.length === 0;
}

// Picks the tier for the OFFLINE-FALLBACK reply only (see mockRespond) --
// when the API is reachable, tier is always the server's real decision and
// this selection is never consulted.
function setTier(tier) {
  state.currentTier = tier;
  for (const seg of els.brainToggle.querySelectorAll(".segment")) {
    const active = seg.dataset.tier === tier;
    seg.classList.toggle("active", active);
    seg.setAttribute("aria-checked", String(active));
  }
  els.emptyStateAvatar.dataset.tier = tier;
  // Keep any engaged composer mode tinted with the tier the avatar is showing.
  for (const btn of [els.videoModeBtn, els.voiceModeBtn]) btn.dataset.tier = tier;
}

els.newChatBtn.addEventListener("click", () => {
  state.activeChatId = null;
  showEmptyState();
  renderChatList();
  els.composerInput.focus();
});

els.searchInput.addEventListener("input", (e) => {
  state.searchQuery = e.target.value.trim();
  renderChatList();
});

els.composer.addEventListener("submit", handleSend);
els.composerInput.addEventListener("input", () => {
  autoGrow();
  updateSendState();
});
els.composerInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    els.composer.requestSubmit();
  }
});

els.brainToggle.addEventListener("click", (e) => {
  const btn = e.target.closest(".segment");
  if (btn) setTier(btn.dataset.tier);
});

els.attachFileBtn.addEventListener("click", () => els.fileInput.click());
els.attachImageBtn.addEventListener("click", () => els.imageInput.click());
els.attachVideoBtn.addEventListener("click", () => els.videoInput.click());

els.fileInput.addEventListener("change", (e) => addAttachments(e.target.files, "file"));
els.imageInput.addEventListener("change", (e) => addAttachments(e.target.files, "image"));
els.videoInput.addEventListener("change", (e) => addAttachments(e.target.files, "video"));

els.videoModeBtn.addEventListener("click", toggleVideoMode);
els.voiceModeBtn.addEventListener("click", toggleVoice);
els.dictationCancel.addEventListener("click", cancelDictation);
els.dictationConfirm.addEventListener("click", confirmDictation);

els.videomodeClose.addEventListener("click", stopVideoMode);
els.videomodeFlip.addEventListener("click", flipCamera);
els.videomodeShutter.addEventListener("click", captureFrame);

// Release mic/camera if the tab goes away rather than holding them open.
window.addEventListener("pagehide", () => {
  if (state.listening) stopDictation();
  if (state.videoMode) stopVideoMode();
});

// Drag-and-drop onto the composer, same destination as the paperclip.
els.composer.addEventListener("dragover", (e) => e.preventDefault());
els.composer.addEventListener("drop", (e) => {
  e.preventDefault();
  if (!e.dataTransfer?.files?.length) return;
  for (const file of e.dataTransfer.files) {
    addAttachments([file], file.type.startsWith("image/") ? "image" : "file");
  }
});

els.exprButtons.addEventListener("click", (e) => {
  const btn = e.target.closest("button");
  if (btn) playExpression(activePreviewAvatar(), btn.dataset.expr);
});

els.profilerPill.addEventListener("click", (e) => {
  e.stopPropagation();
  toggleProfilerCard();
});
document.addEventListener("click", (e) => {
  if (!els.profiler.contains(e.target)) closeProfilerCard();
});
document.addEventListener("keydown", (e) => {
  if (state.listening) {
    // Esc discards, Enter inserts -- same keys ChatGPT's dictation uses.
    if (e.key === "Escape") {
      e.preventDefault();
      cancelDictation();
      return;
    }
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      confirmDictation();
      return;
    }
  }
  if (state.videoMode && e.key === "Escape") {
    e.preventDefault();
    stopVideoMode();
    return;
  }
  if (e.key === "Escape") closeProfilerCard();
});

initVoice();
initVideoMode();
renderChatList();
showEmptyState();
renderAttachments();
updateSendState();
checkBackend();
