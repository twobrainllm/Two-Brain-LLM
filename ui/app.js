/**
 * Two-Brain Chat -- UI shell.
 *
 * MOCK NOTICE: `mockRespond()` below is the only simulated piece. Everything
 * else (history, search, persistence, rendering) is real. See README.md for
 * what "wiring this for real" means later.
 */

const STORAGE_KEY = "twoBrainChats";
const GROUP_ORDER = ["Today", "Yesterday", "Previous 7 Days", "Previous 30 Days", "Older"];

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
  attachBtn: document.getElementById("attach-btn"),
  attachInput: document.getElementById("attach-input"),
  attachmentPreview: document.getElementById("attachment-preview"),
  attachmentThumb: document.getElementById("attachment-thumb"),
  attachmentName: document.getElementById("attachment-name"),
  attachmentRemove: document.getElementById("attachment-remove"),
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
 * "Thinking" choreography for the mock reply delay -- not a real signal,
 * just personality. Cloud gets a longer, more deliberate look-around
 * (bigger brain, harder problem); local gets a quick glance. Real wiring
 * (see README.md) would replace this whole rhythm with a genuine
 * "waiting on the model" state, which has no natural sub-beats to loop
 * through.
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

const state = {
  chats: loadChats(),
  activeChatId: null,
  currentTier: "local",
  //: {dataUrl, name} for the image staged on the composer, or null.
  attachment: null,
  //: null until /api/health answers. When the backend is absent we fall
  //: back to mockRespond() rather than failing -- the UI was built to run
  //: standalone and should keep doing so.
  backend: null,
  //: True while a reply is streaming; suppresses re-renders that
  //: would destroy the bubble being painted into.
  streaming: false,
  searchQuery: "",
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

/**
 * Persist chats, surviving a full store.
 *
 * localStorage is ~5 MB and throws QuotaExceededError when full. This used to
 * be an unguarded setItem, so once attached images pushed the store over the
 * limit the exception propagated out of handleSend *before* the assistant
 * reply was pushed -- losing the reply and freezing storage at the last good
 * save, which read as "earlier chats disappear".
 *
 * Attachments are stored as small thumbnails rather than full images (see
 * makeThumbnail), so this should be rare; if it still happens, the oldest
 * chats are dropped until the rest fits. Losing the oldest history is bad, but
 * silently losing the conversation in front of you is worse.
 */
function saveChats() {
  for (let attempt = 0; attempt < 12; attempt++) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state.chats));
      return true;
    } catch (err) {
      const isQuota =
        err instanceof DOMException &&
        (err.name === "QuotaExceededError" || err.code === 22 || err.code === 1014);
      if (!isQuota) {
        console.error("saveChats failed", err);
        return false;
      }
      // Drop the oldest chat that is not the one being written to.
      const oldest = state.chats
        .filter((c) => c.id !== state.activeChatId)
        .sort((a, b) => (a.updatedAt || 0) - (b.updatedAt || 0))[0];
      if (!oldest) {
        // Nothing left to evict: strip images from the active chat instead.
        let freed = false;
        for (const chat of state.chats) {
          for (const msg of chat.messages) {
            if (msg.image) {
              delete msg.image;
              freed = true;
            }
          }
        }
        if (!freed) {
          console.error("saveChats: cannot fit chats in localStorage");
          return false;
        }
        continue;
      }
      state.chats = state.chats.filter((c) => c !== oldest);
      console.warn("saveChats: storage full, dropped oldest chat", oldest.title);
    }
  }
  return false;
}

/**
 * Shrink an image to a thumbnail data URL for persistence.
 *
 * The full-resolution image still goes to the model; only this small copy is
 * kept in localStorage and shown in history. A 2 MB photo is ~2.7 MB as base64
 * and would blow the quota on its own.
 */
function makeThumbnail(dataUrl, maxPx = 320) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const scale = Math.min(1, maxPx / Math.max(img.width, img.height));
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.round(img.width * scale));
      canvas.height = Math.max(1, Math.round(img.height * scale));
      canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
      try {
        resolve(canvas.toDataURL("image/jpeg", 0.7));
      } catch {
        resolve(null); // tainted canvas or unsupported type; history just loses the preview
      }
    };
    img.onerror = () => resolve(null);
    img.src = dataUrl;
  });
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

  // An attached image is shown on the message that carried it. It never
  // left the device -- the cloud tier has no vision model -- so this is
  // the only place it is ever displayed.
  if (msg.image) {
    const img = document.createElement("img");
    img.className = "message-image";
    img.src = msg.image;
    img.alt = "Attached image";
    bubble.appendChild(img);
  }

  if (msg.role === "assistant") {
    const badge = document.createElement("div");
    badge.className = "tier-badge";
    const tier = msg.tier || "local";
    badge.innerHTML =
      `<span class="tier-dot" data-tier="${tier}"></span>` +
      (tier === "local" ? "Local brain" : "Cloud brain");
    bubble.appendChild(badge);
  }

  const text = document.createElement("div");
  // Assistant replies arrive as Markdown. renderMarkdown escapes the model
  // output before applying any pattern, so this is not an injection point.
  // User messages stay literal: what someone typed should be shown as typed,
  // not reinterpreted.
  if (msg.role === "assistant") {
    text.className = "markdown";
    text.innerHTML = renderMarkdown(msg.content);
  } else {
    text.innerHTML = escapeHtml(msg.content).replace(/\n/g, "<br>");
  }
  bubble.appendChild(text);

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
  return tier === "cloud" ? "Cloud" : "Local";
}

function formatPrivacy(metrics) {
  if (!metrics.piiCount) return "No PII detected";
  const parts = metrics.piiEntities.map((e) => `${e.count} ${e.type}`);
  return `${metrics.piiCount} masked (${parts.join(", ")})`;
}

/** Renders the profiler pill + card from one message's computed metrics (see profiler.js). */
function renderProfiler(metrics, tier) {
  els.profiler.dataset.hasData = "true";
  els.profilerDot.dataset.tier = tier;
  els.profilerCardDot.dataset.tier = tier;
  els.profilerCardTier.textContent = `${tierLabel(tier)} brain`;

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
 * MOCK: stands in for TwoBrainRouter.route(query, context). The tier is
 * whatever the sidebar toggle is set to, not a real routing decision -- see
 * README.md for the real contract this needs to match.
 */

/* ── Backend bridge ───────────────────────────────────────────────────────
   `ui/server.py` exposes the real TwoBrainRouter. When it is not running the
   UI degrades to mockRespond() so the front-end still demos standalone --
   that was true before this backend existed and stays true now.

   The local/cloud switch is only authoritative when the server reports
   ui_test:true (route(force_tier=...) is gated on UI_TEST=1). Otherwise the
   policy decides and the toggle is a *preference*, which the status line
   says out loud rather than pretending otherwise. */
const MAX_IMAGE_BYTES = 12 * 1024 * 1024;

async function probeBackend() {
  try {
    const res = await fetch("/api/health", { method: "GET" });
    if (!res.ok) throw new Error(`health ${res.status}`);
    state.backend = await res.json();
  } catch {
    state.backend = null;
  }
  renderBackendStatus();
}

function renderBackendStatus() {
  let el = document.getElementById("backend-status");
  if (!el) {
    el = document.createElement("div");
    el.id = "backend-status";
    el.className = "backend-status";
    document.querySelector(".sidebar-footer")?.prepend(el);
  }
  const hint = document.getElementById("empty-state-hint");
  const toggleLabel = document.getElementById("brain-toggle-label");
  const b = state.backend;
  if (!b) {
    el.className = "backend-status mock";
    el.innerHTML = '<span class="dot"></span>Mock mode — no backend. Run <code>ui/server.py</code>.';
    if (hint) hint.textContent =
      "Replies are simulated — no backend is running. Start ui/server.py for real answers.";
    if (toggleLabel) toggleLabel.textContent = "Simulated brain";
    els.attachBtn.disabled = true;
    els.attachBtn.title = "Attaching needs the backend (ui/server.py)";
    return;
  }
  el.className = "backend-status live";
  const bits = [`${b.fast_brain} / ${b.deep_brain}`];
  if (!b.vision) bits.push("no vision");
  if (!b.ui_test) bits.push("UI_TEST off — policy decides, switch is a preference");
  el.innerHTML = `<span class="dot"></span>Live — ${bits.join(" · ")}`;
  if (hint) {
    hint.textContent = b.ui_test
      ? `Answers are real. The ${b.vision ? "local VLM" : "local brain"} and the cloud deep brain are both live — pick one in the sidebar.`
      : "Answers are real. The router decides which brain replies; the sidebar switch is only a preference (UI_TEST is off).";
  }
  if (toggleLabel) toggleLabel.textContent = b.ui_test ? "Answering brain" : "Preferred brain (policy decides)";
  els.attachBtn.disabled = !b.vision;
  els.attachBtn.title = b.vision
    ? "Attach an image (stays on-device)"
    : "The local brain has no vision model loaded";
}

async function askBackend(query, tier, imageDataUrl) {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: query, tier, image: imageDataUrl || undefined }),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `backend returned ${res.status}`);
  return body;
}

/**
 * Stream an answer, calling `onDelta(fullTextSoFar)` as tokens arrive.
 *
 * The local brain decodes at ~21 tok/s, so a long reply is a minute of blank
 * screen without this. Resolves to the same shape askBackend returns, so the
 * caller's bookkeeping is unchanged.
 *
 * EventSource is not usable here: it is GET-only, and the request carries a
 * JSON body with an optional base64 image. So this reads the fetch body as a
 * stream and parses the SSE frames directly.
 */
async function askBackendStreaming(query, tier, imageDataUrl, onDelta) {
  const res = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: query, tier, image: imageDataUrl || undefined }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `backend returned ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  const result = { answer: "" };
  let buffer = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line; the last piece may be partial.
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
        continue;
      }

      if (event === "delta") {
        result.answer += payload.text || "";
        onDelta(result.answer);
      } else if (event === "meta") {
        Object.assign(result, payload);
      } else if (event === "done") {
        Object.assign(result, payload, { answer: result.answer });
      } else if (event === "error") {
        throw new Error(payload.error || "stream failed");
      }
    }
  }
  return result;
}

/* ── Image attachment ─────────────────────────────────────────────────── */
function clearAttachment() {
  state.attachment = null;
  els.attachInput.value = "";
  els.attachmentPreview.hidden = true;
  els.attachBtn.classList.remove("has-image");
  updateSendState();
}

function setAttachment(file) {
  if (!file) return;
  if (file.size > MAX_IMAGE_BYTES) {
    alert(`That image is ${(file.size / 1e6).toFixed(1)} MB; the limit is 12 MB.`);
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    state.attachment = { dataUrl: String(reader.result), name: file.name };
    els.attachmentThumb.src = state.attachment.dataUrl;
    els.attachmentName.textContent = file.name;
    els.attachmentPreview.hidden = false;
    els.attachBtn.classList.add("has-image");
    updateSendState();
  };
  reader.readAsDataURL(file);
}

els.attachBtn.addEventListener("click", () => els.attachInput.click());
els.attachInput.addEventListener("change", (e) => setAttachment(e.target.files?.[0]));
els.attachmentRemove.addEventListener("click", clearAttachment);

// Paste an image straight into the composer, as Claude does.
els.composerInput.addEventListener("paste", (e) => {
  if (els.attachBtn.disabled) return;
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (item) {
    e.preventDefault();
    setAttachment(item.getAsFile());
  }
});

function mockRespond(query, tier) {
  const tierLabel = tier === "cloud" ? "Cloud AI 100 (simulated)" : "local fast brain (simulated)";
  return (
    `This is a simulated reply from the ${tierLabel}.\n\n` +
    `Mock mode: no model ran and no query left this browser tab. ` +
    `Flip the "Simulated brain" toggle in the sidebar before sending to preview the other tier's color.`
  );
}

async function handleSend(e) {
  e.preventDefault();
  const query = els.composerInput.value.trim();
  if (!query) return;

  let chat = state.chats.find((c) => c.id === state.activeChatId);
  if (!chat) chat = createChat(query);

  const attachment = state.attachment;
  // The full-resolution image goes to the model; only a thumbnail is kept in
  // the chat, because localStorage cannot hold full images (see saveChats).
  const userMessage = { role: "user", content: query, timestamp: Date.now() };
  chat.messages.push(userMessage);
  if (attachment?.dataUrl) {
    makeThumbnail(attachment.dataUrl).then((thumb) => {
      if (!thumb) return;
      userMessage.image = thumb;
      saveChats();
      // Re-rendering mid-stream would destroy the bubble being painted into.
      const active = state.chats.find((c) => c.id === state.activeChatId);
      if (!state.streaming && active === chat) renderMessages(chat);
    });
  }
  chat.updatedAt = Date.now();
  saveChats();
  renderMessages(chat);
  renderChatList();

  els.composerInput.value = "";
  clearAttachment();
  autoGrow();
  updateSendState();

  const tier = state.currentTier;
  const thinkingStartedAt = performance.now();
  const thinkingRow = renderMessageEl({ role: "assistant", content: "…", tier });
  const thinkingAvatar = thinkingRow.querySelector(".robot-avatar");
  thinkingAvatar?.classList.add("thinking");
  els.messages.appendChild(thinkingRow);
  scrollToBottom();

  const isLongQuery = query.length > LONG_QUERY_CHARS;
  const surpriseMs = isLongQuery ? EXPRESSION_HOLD_MS.surprised : 0;
  if (isLongQuery) {
    playExpression(thinkingAvatar, "surprised");
    await sleep(surpriseMs);
  }

  const thinkMs = THINK_MS_BY_TIER[tier] ?? THINK_MS_BY_TIER.local;
  const lookBudgetMs = Math.max(thinkMs - surpriseMs, 0);
  await playThinkingLooks(thinkingAvatar, lookBudgetMs, THINK_RHYTHM_MS);

  playExpression(thinkingAvatar, "happy", HAPPY_LEAD_MS);
  await sleep(HAPPY_LEAD_MS);

  // Real backend when it is up; the canned reply only when it is not.
  let answer;
  let answeredTier = tier;
  let metrics;
  // The assistant message is added to the chat *before* streaming and filled in
  // as tokens arrive, so a reload part-way through keeps what was received. A
  // local reply can run for minutes; losing all of it to an accidental refresh
  // would be its own bug.
  const streamingMessage = {
    role: "assistant",
    content: "",
    tier,
    timestamp: Date.now(),
    partial: true,
  };
  chat.messages.push(streamingMessage);
  state.streaming = true;
  let lastPersistAt = 0;

  if (state.backend) {
    try {
      // Paint into the placeholder bubble as tokens arrive, so a long local
      // reply shows progress instead of a blank wait.
      const liveText = thinkingRow.querySelector(".bubble > div:last-child");
      const result = await askBackendStreaming(
        query,
        tier,
        attachment?.dataUrl,
        (soFar) => {
          streamingMessage.content = soFar;
          if (liveText) {
            thinkingAvatar?.classList.remove("thinking");
            liveText.className = "markdown";
            liveText.innerHTML = renderMarkdown(soFar);
            scrollToBottom();
          }
          // Persist at most once a second: saving every token would serialise
          // the whole history on each delta.
          const now = performance.now();
          if (now - lastPersistAt > 1000) {
            lastPersistAt = now;
            chat.updatedAt = Date.now();
            saveChats();
          }
        },
      );
      answer = result.answer;
      // The server reports which brain actually answered. Under UI_TEST=0 the
      // policy may well have overruled the toggle, so trust the response
      // rather than the switch.
      answeredTier = result.tier_answered || tier;
      metrics = computeMetrics(query, answeredTier);
      metrics.difficulty = result.difficulty_score ?? metrics.difficulty;
      metrics.estLatencyMs = result.est_latency_ms ?? metrics.estLatencyMs;
      metrics.estCostUsd = result.est_cost_usd ?? metrics.estCostUsd;
      metrics.routerNotes = result.notes || [];
      metrics.piiCount = result.pii_entities_masked ?? metrics.piiCount;
    } catch (err) {
      answer = `The backend returned an error:

${err.message}`;
      metrics = computeMetrics(query, tier);
    }
  } else {
    answer = mockRespond(query, tier);
    metrics = computeMetrics(query, tier);
  }
  metrics.actualLatencyMs = performance.now() - thinkingStartedAt;
  // Finalise the message that was streamed into, rather than pushing a second
  // one -- it is already in chat.messages.
  streamingMessage.content = answer;
  streamingMessage.tier = answeredTier;
  streamingMessage.metrics = metrics;
  delete streamingMessage.partial;
  state.streaming = false;
  chat.updatedAt = Date.now();
  saveChats();
  renderMessages(chat);
  renderChatList();
  playExpression(els.messages.querySelector(".message.assistant:last-child .robot-avatar"), "happy");
  renderProfiler(metrics, answeredTier);
}

function autoGrow() {
  els.composerInput.style.height = "auto";
  els.composerInput.style.height = Math.min(els.composerInput.scrollHeight, 200) + "px";
}

function updateSendState() {
  // An image alone is a valid message; the VLM can be asked to describe it.
  els.sendBtn.disabled =
    els.composerInput.value.trim().length === 0 && !state.attachment;
}

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
// Decides live-vs-mock, whether the tier switch is authoritative, and
// whether the attach button is usable.
probeBackend();
