/* Minimal Markdown renderer for model output.
 *
 * The models reply in Markdown -- headings, **bold**, lists, fenced code --
 * and the UI was rendering it literally, so answers arrived full of asterisks
 * and hashes.
 *
 * No dependency, matching the rest of this UI (no build step, no CDN).
 *
 * SECURITY: model output is untrusted. Everything is HTML-escaped *first*,
 * then a fixed set of Markdown patterns is applied to the escaped text, so a
 * reply containing <script> or an onerror= attribute renders as visible text
 * rather than executing. Link hrefs are additionally restricted to http/https/
 * mailto, which blocks javascript: URLs.
 *
 * `mdEscapeHtml` is named to avoid clobbering app.js's global `escapeHtml`;
 * both are classic scripts sharing one namespace.
 *
 * Plain script, not a module: profiler.js and app.js are loaded the same way
 * and the page has no build step.
 */

function mdEscapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function safeHref(url) {
  const trimmed = url.trim();
  return /^(https?:|mailto:)/i.test(trimmed) ? trimmed : "#";
}

/** Inline spans, applied to already-escaped text. */
function renderInline(text) {
  return (
    text
      // `code` first: its contents must not be re-processed for emphasis.
      .replace(/`([^`\n]+)`/g, (_, c) => `<code>${c}</code>`)
      .replace(/\[([^\]\n]+)\]\(([^)\s]+)\)/g, (_, label, url) =>
        `<a href="${safeHref(url)}" target="_blank" rel="noopener noreferrer">${label}</a>`)
      .replace(/\*\*\*([^*\n]+)\*\*\*/g, "<strong><em>$1</em></strong>")
      .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
      // Single * for italics, but not the bullet at a line start.
      .replace(/(^|[^*\w])\*([^*\n]+)\*(?![*\w])/g, "$1<em>$2</em>")
      .replace(/(^|\s)_([^_\n]+)_(?=\s|$|[.,!?])/g, "$1<em>$2</em>")
      .replace(/~~([^~\n]+)~~/g, "<del>$1</del>")
  );
}

/**
 * Render Markdown to HTML.
 *
 * Line-based rather than a real parser: enough for chat replies, and small
 * enough to audit. An unterminated code fence (which happens constantly while
 * streaming) is closed implicitly at the end.
 */
function renderMarkdown(src) {
  const lines = mdEscapeHtml(String(src ?? "")).split("\n");
  const out = [];
  let listType = null;      // "ul" | "ol" | null
  let inCode = false;
  let paragraph = [];

  const flushParagraph = () => {
    if (paragraph.length) {
      out.push(`<p>${renderInline(paragraph.join(" "))}</p>`);
      paragraph = [];
    }
  };
  const closeList = () => {
    if (listType) {
      out.push(`</${listType}>`);
      listType = null;
    }
  };

  for (const raw of lines) {
    const line = raw.replace(/\s+$/, "");

    // Fenced code. Everything inside is emitted verbatim (already escaped).
    const fence = line.match(/^\s*```(\w*)\s*$/);
    if (fence) {
      if (inCode) {
        out.push("</code></pre>");
        inCode = false;
      } else {
        flushParagraph();
        closeList();
        const lang = fence[1] ? ` class="lang-${fence[1]}"` : "";
        out.push(`<pre><code${lang}>`);
        inCode = true;
      }
      continue;
    }
    if (inCode) {
      out.push(raw + "\n");
      continue;
    }

    if (!line.trim()) {
      flushParagraph();
      closeList();
      continue;
    }

    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      flushParagraph();
      closeList();
      const level = heading[1].length;
      out.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
      continue;
    }

    if (/^\s*([-*_])\1{2,}\s*$/.test(line)) {
      flushParagraph();
      closeList();
      out.push("<hr>");
      continue;
    }

    const quote = line.match(/^\s*&gt;\s?(.*)$/); // ">" is escaped by now
    if (quote) {
      flushParagraph();
      closeList();
      out.push(`<blockquote>${renderInline(quote[1])}</blockquote>`);
      continue;
    }

    const bullet = line.match(/^\s*[-*+]\s+(.*)$/);
    if (bullet) {
      flushParagraph();
      if (listType !== "ul") {
        closeList();
        out.push("<ul>");
        listType = "ul";
      }
      out.push(`<li>${renderInline(bullet[1])}</li>`);
      continue;
    }

    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (numbered) {
      flushParagraph();
      if (listType !== "ol") {
        closeList();
        out.push("<ol>");
        listType = "ol";
      }
      out.push(`<li>${renderInline(numbered[1])}</li>`);
      continue;
    }

    closeList();
    paragraph.push(line.trim());
  }

  flushParagraph();
  closeList();
  if (inCode) out.push("</code></pre>"); // unterminated fence, common mid-stream
  return out.join("");
}
