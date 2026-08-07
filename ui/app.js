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

/** How much prior conversation is sent back with each query.
 *
 * The router is deliberately stateless -- `api.py` shares ONE `TwoBrainRouter`
 * across every request and every browser tab (the NPU's Genie session is far
 * too expensive to rebuild per request), so history kept there would leak
 * between unrelated chats. The client owns it instead, and this is where the
 * bound lives.
 *
 * Bounded on three axes because an unbounded history is expensive in three
 * ways: the on-device model's prompt grows (it already takes 4-12s per query
 * and its compiled context length is finite), and on an escalation the history
 * is masked and sent to the cloud -- so every extra turn is both latency and
 * privacy surface. `compress_context` truncates to 800 chars server-side
 * anyway; keeping the client's budget near that avoids shipping text that is
 * only going to be thrown away.
 */
const HISTORY_MAX_MESSAGES = 6; // ~3 turns
const HISTORY_MAX_CHARS_PER_MESSAGE = 300;
const HISTORY_MAX_CHARS_TOTAL = 1200;

/** Where `two_brain_router.api` is listening.
 *
 * Overridable with `?api=http://host:port` so a second router can be pointed
 * at without editing this file -- two servers on different ports (comparing
 * `--tier pc` against `--tier mobile`, or a new build against a running one),
 * or a LAN demo where the API is on another machine. The default is the
 * loopback address `api.py` binds by default, so the common case needs no
 * query string at all.
 */
const API_BASE_URL = (() => {
  const override = new URLSearchParams(location.search).get("api");
  if (override) return override.replace(/\/$/, "");

  const { protocol, hostname } = location;

  // Opened off the filesystem: no host to borrow, so the loopback default.
  if (protocol === "file:" || !hostname) return "http://127.0.0.1:8765";

  // Served over HTTPS means ui/serve_https.py, which proxies the API on its
  // own origin. Use that origin: an HTTPS page may not fetch an http:// URL
  // at all (mixed content), so naming :8765 directly would be blocked by the
  // browser no matter which host it pointed at.
  if (protocol === "https:") return "";

  // Plain HTTP from another device -- a phone on the LAN. 127.0.0.1 would
  // mean *the phone*, which is not running the router, so borrow the host
  // this page came from and keep api.py's port.
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
  composerInput: document.getElementById("composer-input"),
  sendBtn: document.getElementById("send-btn"),
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
  emptyStateHint: document.getElementById("empty-state-hint"),
  brainToggleLabel: document.getElementById("brain-toggle-label"),
  attachBtn: document.getElementById("attach-btn"),
  attachInput: document.getElementById("attach-input"),
  attachmentPreview: document.getElementById("attachment-preview"),
  attachmentThumb: document.getElementById("attachment-thumb"),
  attachmentName: document.getElementById("attachment-name"),
  attachmentRemove: document.getElementById("attachment-remove"),
  brainToggleWrap: document.getElementById("brain-toggle-wrap"),
  expressionPreviewWrap: document.getElementById("expression-preview-wrap"),
  modelPicker: document.getElementById("model-picker"),
  modelSelect: document.getElementById("model-select"),
  modelDetail: document.getElementById("model-detail"),
};

/** Ceiling on an attached image, matching `api.py`'s `MAX_IMAGE_CHARS`.
 *
 * Checked here as well as server-side so an oversized file is refused before
 * it is base64'd and pushed over the wire, rather than after. Base64 inflates
 * by ~4/3, hence the ratio. */
const MAX_IMAGE_BYTES = 3_300_000;

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
    // `avatarEl` may be a getter rather than an element: on a split, the local
    // bubble finalises mid-flight and a *new* pending row opens for the cloud,
    // so the avatar that should be animating changes while this loop runs.
    // Resolving each beat lets the animation follow it instead of drumming its
    // fingers on a bubble that has already been answered.
    const el = typeof avatarEl === "function" ? avatarEl() : avatarEl;
    playExpression(el, order[i % order.length], rhythmMs);
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

/** Badge text per message tier.
 *
 * A split turn is rendered as **two messages** -- the local answer, then the
 * cloud's completion beneath it -- so `local` and `cloud` are what new messages
 * ever carry. Two bubbles beat one merged bubble here because the whole point
 * of this architecture is that two different models answered two different
 * parts, and a single blob with a combined label hides exactly that.
 *
 * `hybrid` survives for two reasons: chats saved before the split-render change
 * still hold merged messages, and the profiler still describes the *turn* as a
 * whole (where "Local + Cloud" is the accurate summary).
 */
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
  attachment: null,  // {name, dataUrl} while one is staged for the next send
  models: [],        // from GET /models -- only what is installed
  modelId: null,     // the selected local model, sent with every query
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
  return chat.messages.some((m) => m.content.toLowerCase().includes(q));
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

  // The boundary-crossing disclosure is part of the message, not just the live
  // row: this function rebuilds the whole transcript at the end of every turn
  // (and on load), so anything only painted onto the in-flight row vanishes the
  // moment the answer lands. It was doing exactly that before this.
  if (msg.role === "assistant" && msg.crossing) {
    renderCrossing(bubble, msg.crossing);
  }

  // A user message can carry an image. Shown inline so the transcript records
  // what was actually asked -- and it never left this machine: `api.py` decodes
  // it to a temp file the local VLM reads, and only a masked *textual*
  // description is ever eligible to cross the boundary.
  if (msg.role === "user" && msg.image) {
    const img = document.createElement("img");
    img.className = "message-image";
    img.src = msg.image;
    img.alt = "Attached image";
    bubble.appendChild(img);
  }

  const text = document.createElement("div");
  // Assistant replies arrive as Markdown -- headings, **bold**, numbered lists,
  // fenced code -- and were rendering literally, so cloud answers arrived full
  // of asterisks and hashes. `renderMarkdown` (markdown.js) escapes the model's
  // output *first* and only then applies a fixed pattern set, so untrusted
  // output still cannot inject HTML.
  //
  // User messages stay plain on purpose: the user typed them, and silently
  // reformatting someone's own words is worse than showing them verbatim.
  if (msg.role === "assistant" && typeof renderMarkdown === "function") {
    text.className = "markdown";
    text.innerHTML = renderMarkdown(msg.content ?? "");
  } else {
    text.innerHTML = escapeHtml(msg.content ?? "").replace(/\n/g, "<br>");
  }
  bubble.appendChild(text);

  row.appendChild(bubble);
  return row;
}

/**
 * Retags a live row as belonging to a different brain -- badge text, badge dot
 * and avatar colour together.
 *
 * Needed because a row's tier is not always known when it is created: a turn
 * opens as a blue "local" pending row, and only the `escalating` progress event
 * reveals that the local model produced nothing and the cloud is answering
 * instead. Repainting beats guessing, and beats leaving it blue while the cloud
 * works.
 */
/**
 * Renders the "what crossed the boundary" disclosure into a bubble.
 *
 * Shows the masked text that was actually sent, plus each substitution as
 * `typed -> placeholder`. The raw values are the user's own words, already on
 * their screen -- and showing them beside the placeholders is the whole point:
 * a masked string on its own proves nothing without what it replaced.
 *
 * Inserted above whatever else the bubble holds so it stays visible as the
 * answer streams in beneath it.
 */
function renderCrossing(container, payload) {
  // Takes the `.bubble` element itself rather than the live-bubble state
  // object, because `renderMessageEl` builds a bubble *before* attaching it to
  // its row -- a row-based lookup finds nothing there and silently renders
  // none of this.
  if (!container || !payload) return;
  if (container.querySelector && container.querySelector(".crossing")) return;
  const subs = payload.substitutions || [];
  const rows = subs
    .map((s) =>
      `<div class="crossing-sub"><code>${escapeHtml(s.value)}</code>` +
      `<span class="crossing-arrow">-></span>` +
      `<code class="crossing-mask">${escapeHtml(s.placeholder)}</code></div>`
    )
    .join("");

  const el = document.createElement("details");
  el.className = "crossing";
  // The chevron is the only thing telling anyone this opens: the native
  // <details> marker is hidden in CSS (it renders inconsistently across
  // browsers and clashes with the lock), so without this the row is silently
  // clickable -- which is the same as not being clickable at all. It rotates
  // 180 degrees on open, so the icon also reports the current state.
  el.innerHTML =
    `<summary><span class="crossing-lock">&#128274;</span>` +
    `<span class="crossing-summary-text">` +
    (subs.length
      ? `${subs.length} item${subs.length === 1 ? "" : "s"} masked before leaving this device`
      : `Sent to the cloud (no PII found)`) +
    `</span>` +
    `<svg class="crossing-chevron" viewBox="0 0 20 20" width="12" height="12" fill="none" aria-hidden="true">` +
    `<path d="M5 8l5 5 5-5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>` +
    `</summary>` +
    (rows ? `<div class="crossing-subs">${rows}</div>` : "") +
    `<div class="crossing-label">Sent off-device:</div>` +
    `<pre class="crossing-text">${escapeHtml(payload.query || "")}</pre>` +
    (payload.context
      ? `<div class="crossing-label">With context:</div>` +
        `<pre class="crossing-text">${escapeHtml(payload.context)}</pre>`
      : "");

  const badge = container.querySelector ? container.querySelector(".tier-badge") : null;
  if (badge && badge.nextSibling) container.insertBefore(el, badge.nextSibling);
  else container.appendChild(el);
}

function setRowTier(row, tier) {
  const avatar = row.querySelector(".robot-avatar");
  if (avatar) avatar.dataset.tier = tier;
  const badge = row.querySelector(".tier-badge");
  if (badge) {
    badge.innerHTML =
      `<span class="tier-dot" data-tier="${tier}"></span>` +
      (TIER_BADGE_LABEL[tier] ?? TIER_BADGE_LABEL.local);
  }
}

/** Replaces everything in a row's bubble except its tier badge. */
function replaceBubbleBody(row, nodes) {
  const bubble = row.querySelector(".bubble");
  if (!bubble) return null;
  [...bubble.children].forEach((el) => {
    if (!el.classList.contains("tier-badge")) el.remove();
  });
  for (const node of nodes) bubble.appendChild(node);
  return bubble;
}

/**
 * Turns a pending row into a finished answer from one brain.
 *
 * Used for the local half of a split the moment it arrives, so it reads as a
 * completed message from the local brain rather than as a half-drawn one --
 * which is the point of showing it early at all.
 */
function finalizeAssistantRow(row, { tier, content }) {
  row.querySelector(".robot-avatar")?.classList.remove("thinking");
  setRowTier(row, tier);
  const text = document.createElement("div");
  // Same Markdown treatment as a persisted message, so the local half looks
  // identical live and after the turn is saved and re-rendered.
  if (typeof renderMarkdown === "function") {
    text.className = "markdown";
    text.innerHTML = renderMarkdown(content ?? "");
  } else {
    text.innerHTML = escapeHtml(content ?? "").replace(/\n/g, "<br>");
  }
  replaceBubbleBody(row, [text]);
}

/**
 * The inner markup of a "waiting on the cloud" placeholder.
 *
 * Shared by `renderCloudPending` (which owns a whole row) and the live
 * streaming path (which writes into a bubble it is already holding a reference
 * to, and so cannot use `replaceBubbleBody` without dropping that reference).
 * One function so the two cannot drift into wording the same wait differently.
 *
 * Leads with "Thinking" in both branches, because that word is the part doing
 * the work: this is shown the instant the router *decides* to cross, which can
 * be many seconds before the cloud emits its first token. `gap` is the
 * interesting detail when there is one -- it names exactly what the local model
 * could not do, and therefore exactly what is crossing the boundary. With no
 * gap the whole query is going, and this says that rather than implying a split
 * that isn't happening.
 */
function cloudPendingHtml(gap) {
  return (
    `<span class="cloud-pending-dots"><i></i><i></i><i></i></span>` +
    `<span class="cloud-pending-text">` +
    (gap
      ? `Thinking… answering the rest: ${escapeHtml(gap)}`
      : `Thinking… escalating the whole query to the cloud`) +
    `</span>`
  );
}

/** Marks a row as waiting on the cloud, optionally naming the gap being asked. */
function renderCloudPending(row, gap) {
  setRowTier(row, "cloud");
  const pending = document.createElement("div");
  pending.className = "cloud-pending";
  pending.innerHTML = cloudPendingHtml(gap);
  replaceBubbleBody(row, [pending]);
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
    body: JSON.stringify({ query, context, model: state.modelId || undefined }),
  });
  const body = await res.json();
  if (!res.ok) throw new Error(body.error || `HTTP ${res.status}`);
  return body;
}

/* ── Image attachment ───────────────────────────────────────────────────── */

function clearAttachment() {
  state.attachment = null;
  if (els.attachInput) els.attachInput.value = "";
  if (els.attachmentPreview) els.attachmentPreview.hidden = true;
  els.attachBtn?.classList.remove("has-image");
  updateSendState();
}

/**
 * Reads a chosen file into a data URL and shows a preview.
 *
 * The image never leaves this machine: `api.py` decodes it to a temp file, the
 * local VLM reads it, and — if the query escalates — only a *masked textual
 * description* crosses the boundary, never the pixels. The cloud tier has no
 * vision model at all, so there is nowhere for an image to go even if we
 * wanted to send one. The preview says so, because "where did my photo go" is
 * the first thing anyone should be able to answer.
 */
function setAttachment(file) {
  if (!file) return;
  if (file.size > MAX_IMAGE_BYTES) {
    alert(
      `That image is ${(file.size / 1e6).toFixed(1)} MB; the limit is ` +
      `${(MAX_IMAGE_BYTES / 1e6).toFixed(1)} MB.`
    );
    clearAttachment();
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    state.attachment = { name: file.name, dataUrl: String(reader.result) };
    if (els.attachmentThumb) els.attachmentThumb.src = state.attachment.dataUrl;
    if (els.attachmentName) els.attachmentName.textContent = file.name;
    if (els.attachmentPreview) els.attachmentPreview.hidden = false;
    els.attachBtn?.classList.add("has-image");
    updateSendState();
  };
  reader.readAsDataURL(file);
}

/**
 * Token-level streaming over Server-Sent Events (`POST /route/sse`).
 *
 * `onEvent(kind, payload)` is called per frame:
 *
 *   meta  {tier, streaming}      once, before any text
 *   delta {text, tier}           repeatedly, tier-attributed
 *   tier  {tier, gap}            a split opening the cloud's bubble
 *   done  {...RouteDecision}     once
 *   error {error}                in-band, since the status line is long sent
 *
 * `EventSource` cannot be used: it is GET-only, and this request carries a JSON
 * body with an optional base64 image. So the `fetch` body is read as a stream
 * and the frames parsed by hand.
 *
 * **Only the local model's `solution` field arrives as deltas** — the router
 * strips the surrounding JSON before it ever reaches the wire, so a bubble is
 * never seen filling with `{"solution": "`.
 *
 * Frames are separated by a blank line and split across chunk boundaries
 * arbitrarily, so `buffer` holds the incomplete tail between reads.
 */
async function sendToRouterSSE(query, context, imageDataUrl, onEvent, signal) {
  const res = await fetch(`${API_BASE_URL}/route/sse`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      context,
      image: imageDataUrl || undefined,
      // Sent every time, so the server cannot drift out of sync with the
      // dropdown. An unchanged id is a no-op server-side.
      model: state.modelId || undefined,
    }),
    signal,
  });
  if (res.status === 404) {
    // A server older than this endpoint. Degrade to the non-streaming path
    // rather than to the offline preview -- the API *is* reachable, and
    // claiming "no model ran" about a working router is a lie that sends
    // someone hunting the wrong problem. (This is exactly what happened: a
    // stale server kept port 8765, the new one failed to bind, and the UI
    // reported the backend as down.)
    console.warn("/route/sse not found — falling back to /route/stream");
    const body = await sendToRouterStreaming(query, context, (progress) => {
      if (progress.phase === "local_answer") {
        onEvent("delta", { text: progress.local_answer, tier: "local" });
        if (progress.gap) onEvent("tier", { tier: "cloud", gap: progress.gap });
      }
    });
    // The pre-SSE endpoints deliver whole halves, not tokens, so emit whatever
    // the deltas above did not already cover.
    if (body.tier_answered === "hybrid") {
      onEvent("delta", { text: body.cloud_answer, tier: "cloud" });
    } else {
      onEvent("delta", { text: body.answer, tier: body.tier_answered });
    }
    onEvent("done", body);
    return body;
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `HTTP ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result = null;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      let event = "message";
      let data = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (!data) continue;
      let payload;
      try {
        payload = JSON.parse(data);
      } catch {
        continue; // keep-alive or a partial frame; not fatal
      }
      if (event === "error") throw new Error(payload.error || "stream failed");
      if (event === "done") result = payload;
      onEvent(event, payload);
    }
  }
  if (!result) throw new Error("stream ended without a result");
  return result;
}

/**
 * Streaming call: POSTs to `/route/stream` and resolves with the final result,
 * calling `onProgress` with the local half as soon as the server has it.
 *
 * Why bother: the local model answers in ~4s and the cloud gap-fill has been
 * measured at 15s+, so the non-streaming `/route` spends most of its wall-clock
 * sitting on an answer that was ready the whole time. This shows that answer
 * immediately and lets the cloud half land when it lands.
 *
 * NDJSON, one JSON object per line. Lines can be split across chunk
 * boundaries, so `buf` holds the incomplete tail between reads -- parsing per
 * chunk instead would fail intermittently on exactly the long answers this
 * feature exists for.
 *
 * Falls back to non-streaming `/route` on 404, so a UI newer than its server
 * degrades to "slower" rather than to the offline preview, which would wrongly
 * claim nothing ran.
 */
async function sendToRouterStreaming(query, context, onProgress) {
  const res = await fetch(`${API_BASE_URL}/route/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, context, model: state.modelId || undefined }),
  });
  if (res.status === 404) return sendToRouter(query, context);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let result = null;

  const handleLine = (line) => {
    if (!line.trim()) return;
    const msg = JSON.parse(line);
    if (msg.type === "progress") onProgress?.(msg);
    else if (msg.type === "result") result = msg;
    else if (msg.type === "error") throw new Error(msg.error);
  };

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop(); // keep the (possibly incomplete) tail for the next read
    for (const line of lines) handleLine(line);
  }
  if (buf.trim()) handleLine(buf);

  if (!result) throw new Error("stream ended without a result");
  return result;
}

/**
 * Builds the `context` string sent with a query: the recent turns of *this*
 * chat, oldest-first, so "explain in more detail" has something to refer to.
 *
 * `excludeLast` drops the message just pushed -- the current query is sent
 * separately as `query`, and repeating it in the context would have the local
 * model answering it twice.
 *
 * Truncation is oldest-first (`slice(-N)`) because recency is what resolves a
 * pronoun. The per-message cap keeps one long answer from crowding out the
 * turns around it, which is the case that actually breaks reference
 * resolution -- a single 4000-char reply would otherwise be the entire budget.
 */
function buildConversationContext(chat, excludeLast = true) {
  if (!chat) return "";
  let msgs = chat.messages.filter((m) => m.role === "user" || m.role === "assistant");
  if (excludeLast) msgs = msgs.slice(0, -1);
  msgs = msgs.slice(-HISTORY_MAX_MESSAGES);
  if (!msgs.length) return "";

  const lines = msgs.map((m) => {
    const who = m.role === "user" ? "User" : "Assistant";
    let body = (m.content || "").replace(/\s+/g, " ").trim();
    if (body.length > HISTORY_MAX_CHARS_PER_MESSAGE) {
      body = body.slice(0, HISTORY_MAX_CHARS_PER_MESSAGE) + "…";
    }
    return `${who}: ${body}`;
  });

  let out = `Earlier in this conversation:\n${lines.join("\n")}`;
  if (out.length > HISTORY_MAX_CHARS_TOTAL) {
    // Trim from the front: the newest turns are the ones a follow-up refers to.
    out = "Earlier in this conversation (truncated):\n" +
      out.slice(out.length - HISTORY_MAX_CHARS_TOTAL);
  }
  return out;
}

/**
 * Loads the installed local models and populates the picker.
 *
 * `GET /models` returns only what is actually on disk, so the dropdown never
 * offers something that will fail 17 seconds into a cold load. Hidden entirely
 * when the backend is down or there is nothing to choose between -- a
 * single-entry dropdown is a decoration, not a control.
 */
async function loadModels() {
  try {
    const res = await fetch(`${API_BASE_URL}/models`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const body = await res.json();
    state.models = body.models || [];
    state.modelId = body.active || state.modelId || state.models[0]?.id || null;
  } catch {
    state.models = [];
    state.modelId = null;
  }
  renderModelPicker();
}

function renderModelPicker() {
  if (!els.modelPicker || !els.modelSelect) return;
  const show = state.models.length > 1;
  els.modelPicker.hidden = !show;
  if (!show) return;

  els.modelSelect.innerHTML = state.models
    .map((m) => `<option value="${escapeHtml(m.id)}">${escapeHtml(m.label)}</option>`)
    .join("");
  if (state.modelId) els.modelSelect.value = state.modelId;
  const active = state.models.find((m) => m.id === els.modelSelect.value);
  if (els.modelDetail) els.modelDetail.textContent = active ? active.detail : "";
}

/**
 * Switching model closes the running brain and cold-loads the new one -- 12s
 * for the NPU, ~17s for the 8B. The server does that work on the next query
 * rather than eagerly, so the select is only disabled while a query is
 * actually in flight; here it just records the choice and updates the caption.
 */
function onModelChange() {
  state.modelId = els.modelSelect.value;
  const active = state.models.find((m) => m.id === state.modelId);
  if (els.modelDetail) {
    els.modelDetail.textContent = active
      ? `${active.detail} - first query after a switch pays a cold load`
      : "";
  }
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
  // Reported in the empty state rather than a permanent sidebar box: it is
  // read once, when you are deciding whether to trust the next reply, and a
  // status chip sitting in the corner for the whole session is noise after
  // that. The tier toggle's own label doubles as the reminder, since offline
  // is the only time that toggle does anything.
  if (els.emptyStateHint) {
    els.emptyStateHint.textContent = live
      ? `Connected to the two-brain router — routing as ${tier}.`
      : `Router not reachable at ${API_BASE_URL}. Replies fall back to an offline preview.`;
    els.emptyStateHint.dataset.live = String(live);
  }
  if (els.brainToggleLabel) {
    els.brainToggleLabel.textContent = live
      ? "Preview tier (unused while live)"
      : "Preview tier (offline)";
  }

  // Both sidebar panels are demo controls. When the router is live the tier
  // toggle cannot force anything -- `route()` decides -- and the expression
  // buttons only preview animations. Hidden rather than left present-but-inert:
  // a control that looks live and does nothing is worse than no control. When
  // the backend is *down* the toggle is the only thing that does anything, so
  // it comes back.
  for (const el of [els.brainToggleWrap, els.expressionPreviewWrap]) {
    if (el) el.hidden = Boolean(live);
  }
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

async function handleSend(e) {
  e.preventDefault();
  const query = els.composerInput.value.trim();
  if (!query) return;

  let chat = state.chats.find((c) => c.id === state.activeChatId);
  if (!chat) chat = createChat(query);

  chat.messages.push({
    role: "user",
    content: query,
    image: state.attachment?.dataUrl,
    timestamp: Date.now(),
  });
  chat.updatedAt = Date.now();
  saveChats();
  renderMessages(chat);
  renderChatList();

  els.composerInput.value = "";
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

  // Built from `chat.messages`, which already had the current query pushed --
  // hence `excludeLast`. Read *before* awaiting anything, so a second send
  // while this one is in flight can't fold a half-finished turn into it.
  const attachment = state.attachment;
  clearAttachment();
  const history = buildConversationContext(chat);

  // One bubble per tier that actually speaks, created on that tier's first
  // delta. A split therefore shows the on-device partial in the local colour
  // and the cloud's gap-fill in its own, filling in as the tokens land.
  const bubbles = new Map(); // tier -> {row, textEl, text}
  let pendingRow = thinkingRow; // whichever row is currently animating
  // Kept for the turn, not just the live row: `renderMessages` rebuilds the
  // transcript from `chat.messages` when the turn ends, so anything not
  // persisted onto a message disappears the moment the answer lands.
  let crossing = null;

  const bubbleFor = (tier) => {
    let b = bubbles.get(tier);
    if (b) return b;
    let row;
    if (bubbles.size === 0) {
      row = thinkingRow; // reuse the placeholder already on screen
    } else {
      row = renderMessageEl({ role: "assistant", content: "", tier });
      row.querySelector(".robot-avatar")?.classList.add("thinking");
      els.messages.appendChild(row);
    }
    setRowTier(row, tier);
    const textEl = document.createElement("div");
    textEl.className = "markdown";
    replaceBubbleBody(row, [textEl]);
    b = { row, textEl, text: "" };
    bubbles.set(tier, b);
    pendingRow = row; // the animation follows whichever brain is speaking
    scrollToBottom();
    return b;
  };

  const onEvent = (kind, payload) => {
    if (kind === "delta") {
      const b = bubbleFor(payload.tier || "local");
      b.text += payload.text || "";
      // Re-render the whole bubble each delta rather than appending text: a
      // Markdown document is not append-safe -- a list or fence half-arrived is
      // not valid Markdown, and rendering it incrementally would leave broken
      // structure behind once the rest lands.
      b.textEl.innerHTML =
        typeof renderMarkdown === "function"
          ? renderMarkdown(b.text)
          : escapeHtml(b.text).replace(/\n/g, "<br>");
      scrollToBottom();
    } else if (kind === "crossing") {
      crossing = payload;
      // What actually left the device, shown *while* the cloud is working --
      // the point is to see it during the wait it bought, not as a footnote
      // afterwards. Collapsed by default: it is evidence, not content.
      renderCrossing(bubbleFor("cloud").row.querySelector(".bubble"), payload);
    } else if (kind === "tier" && payload.tier === "cloud") {
      // The router has *decided* to cross -- masking, the boundary assert and
      // the cloud call have not happened yet. Opening the bubble here rather
      // than on the first cloud delta is the whole point: that delta can be 15s
      // away, and until it lands the screen would otherwise show a finished
      // local answer and no sign that anything else is coming.
      //
      // Fires with or without a gap. It used to require one, which meant the
      // "escalating the whole query" case -- the slowest of the two, since the
      // cloud is answering from scratch -- was the one with no indicator at all.
      const b = bubbleFor("cloud");
      if (!b.text) b.textEl.innerHTML = `<span class="cloud-pending">${cloudPendingHtml(payload.gap)}</span>`;
    }
  };

  const routed = sendToRouterSSE(query, history, attachment?.dataUrl, onEvent, null);
  // A getter, not the element: `pendingRow` moves when a split opens the
  // cloud's row, and the animation should follow it there.
  const looping = playThinkingLooksUntilSettled(
    () => pendingRow.querySelector(".robot-avatar"),
    routed,
    THINK_RHYTHM_MS
  );

  let tier, answer, metrics, live, splitAnswers = null;
  // Fallback for the non-streaming path, where no `crossing` event is
  // emitted but the finished decision carries the same payload.
  let body_crossed = null;
  try {
    const body = await routed;
    await looping; // let the current beat finish instead of cutting it off
    tier = body.tier_answered;
    answer = body.answer;
    metrics = metricsFromRouteResponse(body);
    body_crossed = body.crossed_to_cloud || null;
    live = true;
    if (tier === "hybrid") {
      // Two messages, not one: two models answered two different parts, and a
      // single merged bubble hides precisely that.
      splitAnswers = { local: body.local_answer, cloud: body.cloud_answer };
    }
    setBackendStatus(true, body.tier);
  } catch (err) {
    console.warn("two-brain-router API unreachable, using offline preview:", err);
    // `routed` rejects fast (a refused connection, not a real wait), so
    // `looping` already stopped -- swap to the personality-paced budget
    // loop instead of a near-instant reply, same rhythm the mock always had.
    const fallbackTier = state.currentTier;
    await playThinkingLooks(
      pendingRow.querySelector(".robot-avatar"),
      THINK_MS_BY_TIER[fallbackTier] ?? THINK_MS_BY_TIER.local,
      THINK_RHYTHM_MS
    );
    tier = fallbackTier;
    answer = mockRespond(query, tier);
    metrics = computeMetrics(query, tier);
    live = false;
    setBackendStatus(false);
  }

  playExpression(pendingRow.querySelector(".robot-avatar"), "happy", HAPPY_LEAD_MS);
  await sleep(HAPPY_LEAD_MS);

  metrics.actualLatencyMs = performance.now() - thinkingStartedAt;
  const at = Date.now();
  if (splitAnswers) {
    // Metrics ride on the cloud message alone: they describe the whole turn
    // (total latency, total cost, what was masked), and `renderMessages` picks
    // the last assistant message carrying them for the profiler. Duplicating
    // them onto the local half would double-count the turn in that view.
    chat.messages.push({ role: "assistant", content: splitAnswers.local, tier: "local", live, timestamp: at });
    chat.messages.push({
      role: "assistant", content: splitAnswers.cloud, tier: "cloud", live, timestamp: at, metrics,
      // Rides on the cloud message so it survives the re-render at the end of
      // the turn -- and a reload, since it is part of that turn's audit trail.
      crossing: crossing || body_crossed,
    });
  } else {
    // `live === false` means we fell into the catch: the request failed after
    // the local half had already streamed and been read, so it is kept rather
    // than replaced by an offline-preview blob.
    //
    // Only on that path. On success `answer` *is* the local text, and pushing
    // both produced the same reply twice -- once from the stream, once from
    // the final decision.
    const streamedLocal = live === false ? bubbles.get("local")?.text : null;
    if (streamedLocal) {
      chat.messages.push({ role: "assistant", content: streamedLocal, tier: "local", live: true, timestamp: at });
    }
    chat.messages.push({
      role: "assistant", content: answer, tier, live, timestamp: at, metrics,
      crossing: tier === "cloud" ? crossing || body_crossed : undefined,
    });
  }
  chat.updatedAt = at;
  saveChats();
  renderMessages(chat);
  renderChatList();
  playExpression(els.messages.querySelector(".message.assistant:last-child .robot-avatar"), "happy");
  // The profiler describes the *turn*, so a split is still "Local + Cloud"
  // there even though the transcript now shows it as two messages.
  renderProfiler(metrics, tier);
}

function autoGrow() {
  els.composerInput.style.height = "auto";
  els.composerInput.style.height = Math.min(els.composerInput.scrollHeight, 200) + "px";
}

function updateSendState() {
  els.sendBtn.disabled = els.composerInput.value.trim().length === 0;
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

els.exprButtons.addEventListener("click", (e) => {
  const btn = e.target.closest("button");
  if (btn) playExpression(activePreviewAvatar(), btn.dataset.expr);
});

els.modelSelect?.addEventListener("change", onModelChange);
els.attachBtn?.addEventListener("click", () => els.attachInput?.click());
els.attachInput?.addEventListener("change", (e) => setAttachment(e.target.files?.[0]));
els.attachmentRemove?.addEventListener("click", clearAttachment);

els.profilerPill.addEventListener("click", (e) => {
  e.stopPropagation();
  toggleProfilerCard();
});
document.addEventListener("click", (e) => {
  if (!els.profiler.contains(e.target)) closeProfilerCard();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeProfilerCard();
});

renderChatList();
showEmptyState();
updateSendState();
checkBackend();
loadModels();
