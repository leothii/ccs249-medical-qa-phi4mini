/* ═══════════════════════════════════════════════════════════════════════
  MedAI — Chatbot Frontend Logic
  ═══════════════════════════════════════════════════════════════════════ */

const FALLBACK_API = "http://localhost:8000";
const SAME_ORIGIN_API = (window.location.protocol === "http:" || window.location.protocol === "https:")
  ? window.location.origin
  : "";
const DEFAULT_API = SAME_ORIGIN_API || FALLBACK_API;
// Load persisted settings early so API_BASE can be configured
const _persistedSettings = JSON.parse(localStorage.getItem("medai-settings") || "{}");
let API_BASE = _persistedSettings.apiBase || DEFAULT_API;

function persistSettings(overrides = {}) {
  const existingSettings = JSON.parse(localStorage.getItem("medai-settings") || "{}");
  localStorage.setItem("medai-settings", JSON.stringify({
    ...existingSettings,
    ...overrides,
    apiBase: API_BASE,
  }));
}

// ─── State ────────────────────────────────────────────────────────────────────
let isLoading = false;
let history   = JSON.parse(localStorage.getItem("medai-history") || "[]");
let savedAnswers = JSON.parse(localStorage.getItem("medai-saved-answers") || "[]");
const AI_AVATAR_HTML = `<img src="assets/logo-avatar.png" alt="MedAI logo">`;

// ─── DOM Refs ─────────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

const chatMessages   = $("chat-messages");
const chatInput      = $("chat-input");
const chatInputShell = chatInput?.closest(".lg-input");
const attachBtn      = $("attach-btn");
const attachInput    = $("attach-input");
const sendBtn        = $("send-btn");
const statusDot      = $("status-dot");
const statusText     = $("status-text");
const sumInput       = $("sum-input");
const sumBtn         = $("sum-btn");
const sumResult      = $("sum-result");
const sumResultText  = $("sum-result-text");
const rightSumInput  = $("right-sum-input");
const rightSumBtn    = $("right-sum-btn");
const rightSumResult = $("right-sum-result");
const rightSumText   = $("right-sum-result-text");
const historyPanel   = $("history-panel");
const savedPanel     = $("saved-panel");
const toastEl        = $("toast");

const URGENT_PATTERN = /\b(chest pain|trouble breathing|difficulty breathing|shortness of breath|stroke|face droop|fainting|seizure|suicidal|suicide|overdose|severe bleeding|anaphylaxis|severe allergic|blue lips|loss of consciousness|heart attack)\b/i;
const URGENT_WARNING = "Urgent safety note: your question may describe symptoms that need immediate care. If this is happening now, call your local emergency number or go to the nearest emergency department. MedAI can provide general information, but it cannot assess emergencies.";
const REQUEST_TIMEOUT_MS = 120000;

async function fetchWithTimeout(url, options = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
  const ctrl = new AbortController();
  const timeoutId = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: ctrl.signal });
  } finally {
    clearTimeout(timeoutId);
  }
}

// ─── HEALTH CHECK ─────────────────────────────────────────────────────────────
async function checkHealth() {
  // Normalize the base and build a small set of candidate roots to try.
  const normalize = (u) => (u || "").replace(/\/*$/, "");
  const base = normalize(API_BASE);

  const candidates = new Set();
  candidates.add(base);
  // If base ends with /predict, also try the host without it
  if (/\/predict$/i.test(base)) candidates.add(base.replace(/\/predict$/i, ""));
  // If base does not end with /predict, try adding it
  if (!/\/predict$/i.test(base)) candidates.add(base + "/predict");

  const tryFetch = async (candidate) => {
    try {
      const ctrl = new AbortController();
      setTimeout(() => ctrl.abort(), 4000);
      const root = candidate.replace(/\/$/, "");

      // Helper to try a safe request and return truthy info when endpoint exists.
      const probe = async (method, path = "") => {
        try {
          const res = await fetch(root + path, { method, signal: ctrl.signal });
          // Treat 2xx/3xx as success; 405 Method Not Allowed also indicates endpoint exists
          if ((res.status >= 200 && res.status < 400) || res.status === 405) {
            // try to parse json if possible, else return minimal info
            try { return await res.json(); } catch { return { ok: true, status: res.status, path }; }
          }
          return null;
        } catch (e) {
          return null;
        }
      };

      // Only /health contains reliable model status.
      const r = await probe("GET", "/health");
      return r && Object.prototype.hasOwnProperty.call(r, "model_loaded") ? r : null;
    } catch (e) {
      return null;
    }
  };

  let success = null;
  for (const c of candidates) {
    const d = await tryFetch(c);
    if (d) {
      success = { data: d, url: c };
      break;
    }
  }

  if (success) {
    // Persist the working candidate
    API_BASE = success.url;
    persistSettings();
    statusDot.classList.remove("offline");
    statusText.textContent = success.data.model_loaded ? "Online — Model Ready" : "Online (Demo Mode)";
    return;
  }

  statusDot.classList.add("offline");
  statusText.textContent = "Offline — Model server unreachable";
}

// ─── TAB SWITCHING ────────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll(".nav-item[data-tab]").forEach(n => {
    n.classList.toggle("active", n.dataset.tab === name);
  });
  document.querySelectorAll(".tab-panel").forEach(p => {
    p.classList.toggle("active", p.id === `tab-${name}`);
  });
  if (name === "history") renderHistory();
  if (name === "saved") renderSavedAnswers();
}

function setChatQuestion(question, autoSend = false) {
  switchTab("chat");
  chatInput.value = question;
  autoResize(chatInput);
  chatInput.focus();
  if (autoSend) sendMessage();
}

// ─── TEXTAREA AUTOSIZE ────────────────────────────────────────────────────────
function autoResize(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 120) + "px";
}

// ─── ESCAPE / FORMAT MESSAGE BODY ─────────────────────────────────────────────
function formatBody(text) {
  // Escape HTML
  let html = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // Bold **text**
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

  // Bullet lines (•, -, *)
  const lines = html.split("\n");
  let inList = false;
  const out  = [];
  for (const line of lines) {
    const m = line.match(/^\s*[•\-\*]\s+(.+)$/);
    if (m) {
      if (!inList) { out.push("<ul>"); inList = true; }
      out.push(`<li>${m[1]}</li>`);
    } else {
      if (inList) { out.push("</ul>"); inList = false; }
      if (line.trim()) out.push(`<p>${line}</p>`);
    }
  }
  if (inList) out.push("</ul>");
  return out.join("");
}

// ─── APPEND MESSAGE ───────────────────────────────────────────────────────────
function appendMessage(role, content, meta = null) {
  const wrap = document.createElement("div");
  wrap.className = `msg msg--${role}`;

  if (role === "ai") {
    const av = document.createElement("div");
    av.className = "msg__avatar";
    av.innerHTML = AI_AVATAR_HTML;
    wrap.appendChild(av);
  }

  const bubble = document.createElement("div");
  bubble.className = "msg__bubble";

  // Inner content holder
  const inner = document.createElement("div");
  inner.className = "msg__bubble-inner";
  inner.innerHTML = formatBody(content);
  bubble.appendChild(inner);

  // Meta (response estimate + source + copy)
  if (meta && role === "ai") {
    const confClass =
      meta.confidence >= 80 ? "conf-high" :
      meta.confidence >= 55 ? "conf-mid"  : "conf-low";

    const metaDiv = document.createElement("div");
    metaDiv.className = "msg__meta";
    metaDiv.innerHTML = `
      <span class="msg__confidence ${confClass}">
        <span class="conf-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
        </span>
        <span><strong>Response estimate:</strong> ${meta.confidence}%</span>
      </span>
      <span class="msg__source">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2zM22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>
        <span style="color:var(--lg-blue-bright)">${meta.source}</span>
      </span>
      <div class="msg__actions">
        <button class="msg__action" data-action="copy" title="Copy answer">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
        </button>
        <button class="msg__action" data-action="save" title="Save answer">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/></svg>
        </button>
        ${meta.question ? `
        <button class="msg__action" data-action="retry" title="Ask again">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 1 1-2.64-6.36"/><path d="M21 3v6h-6"/></svg>
        </button>` : ""}
      </div>`;
    metaDiv.querySelector('[data-action="copy"]')?.addEventListener("click", () => copyText(content));
    metaDiv.querySelector('[data-action="save"]')?.addEventListener("click", () => saveAnswer(content, meta));
    metaDiv.querySelector('[data-action="retry"]')?.addEventListener("click", () => setChatQuestion(meta.question, true));
    bubble.appendChild(metaDiv);
  }

  // Related questions
  if (meta?.related?.length) {
    const rel = document.createElement("div");
    rel.className = "related-box";
    rel.innerHTML = `
      <div class="related-box__title">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
        Related Questions
      </div>
      <div class="related-pills"></div>`;
    const pillsHost = rel.querySelector(".related-pills");
    meta.related.forEach(q => {
      const pill = document.createElement("button");
      pill.className = "related-pill";
      pill.innerHTML = `${q} <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>`;
      pill.addEventListener("click", () => askRelated(q));
      pillsHost.appendChild(pill);
    });
    bubble.appendChild(rel);
  }

  wrap.appendChild(bubble);
  chatMessages.appendChild(wrap);
  requestAnimationFrame(() => {
    chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: "smooth" });
  });
}

// ─── TYPING INDICATOR ─────────────────────────────────────────────────────────
function showTyping() {
  const div = document.createElement("div");
  div.className = "msg msg--ai";
  div.id = "typing-indicator";
  div.innerHTML = `
    <div class="msg__avatar">
      ${AI_AVATAR_HTML}
    </div>
    <div class="msg__bubble">
      <div class="msg__bubble-inner typing-bubble">
        <div class="heartbeat-loader" aria-label="MedAI is thinking">
          <svg viewBox="0 0 120 32" role="img">
            <defs>
              <linearGradient id="heartbeat-gradient" x1="0" y1="0" x2="120" y2="0" gradientUnits="userSpaceOnUse">
                <stop offset="0" stop-color="#16B0DD"/>
                <stop offset="0.52" stop-color="#974994"/>
                <stop offset="1" stop-color="#FD931D"/>
              </linearGradient>
            </defs>
            <path class="heartbeat-loader__base" d="M2 18H30L37 18L43 6L51 28L60 18H76L82 12L88 18H118"/>
            <path class="heartbeat-loader__pulse" d="M2 18H30L37 18L43 6L51 28L60 18H76L82 12L88 18H118"/>
          </svg>
        </div>
      </div>
    </div>`;
  chatMessages.appendChild(div);
  requestAnimationFrame(() => {
    chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: "smooth" });
  });
}
function removeTyping() {
  document.getElementById("typing-indicator")?.remove();
}

function appendStreamingMessage() {
  const wrap = document.createElement("div");
  wrap.className = "msg msg--ai";
  wrap.id = "streaming-message";

  const av = document.createElement("div");
  av.className = "msg__avatar";
  av.innerHTML = AI_AVATAR_HTML;
  wrap.appendChild(av);

  const bubble = document.createElement("div");
  bubble.className = "msg__bubble";

  const inner = document.createElement("div");
  inner.className = "msg__bubble-inner";
  inner.textContent = "";
  bubble.appendChild(inner);

  wrap.appendChild(bubble);
  chatMessages.appendChild(wrap);
  requestAnimationFrame(() => {
    chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: "smooth" });
  });

  return { wrap, inner };
}

function parseSseBuffer(buffer, onEvent) {
  const parts = buffer.split("\n\n");
  const remainder = parts.pop() || "";
  for (const part of parts) {
    const lines = part.split("\n");
    let event = "message";
    const dataLines = [];
    for (const line of lines) {
      if (line.startsWith("event:")) {
        event = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trimStart());
      }
    }
    if (!dataLines.length) continue;
    onEvent(event, dataLines.join("\n"));
  }
  return remainder;
}

function handleChatData(data, question) {
  const answerText = data.answer || data.response || data.answer_text || data.summary || JSON.stringify(data);
  const confidence = data.confidence || data.confidence_score || null;
  const source     = data.source || data.from || "Remote API";
  const related    = data.related_questions || data.related || [];
  const warning    = data.urgent_warning || data.warning || "";

  appendMessage("ai", answerText, {
    confidence: confidence ?? 0,
    source:     source,
    related:    related,
    question:   question,
  });
  if (warning && !URGENT_PATTERN.test(question)) {
    appendMessage("ai", warning);
  }

  history.unshift({ question, answer: answerText, ts: Date.now() });
  if (history.length > 50) history.pop();
  localStorage.setItem("medai-history", JSON.stringify(history));
}

// ─── SEND MESSAGE ─────────────────────────────────────────────────────────────
async function sendMessage() {
  const question = chatInput.value.trim();
  if (!question || isLoading) return;

  isLoading = true;
  sendBtn.disabled = true;
  chatInput.value = "";
  chatInput.style.height = "auto";
  clearTimeout(typingGlowTimeout);
  chatInputShell?.classList.remove("is-typing");

  appendMessage("user", question);
  if (URGENT_PATTERN.test(question)) {
    appendMessage("ai", URGENT_WARNING);
  }
  showTyping();

  try {
    const res = await fetchWithTimeout(`${API_BASE}/chat_stream`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ question }),
    });

    if (!res.ok || !res.body) {
      if (res.status === 404) {
        const fallback = await fetchWithTimeout(`${API_BASE}/chat`, {
          method:  "POST",
          headers: { "Content-Type": "application/json" },
          body:    JSON.stringify({ question }),
        });
        removeTyping();
        if (!fallback.ok) {
          const err = await fallback.json().catch(() => ({}));
          const detail = err.detail ? `\n\n${err.detail}` : "";
          appendMessage("ai", `Error: ${err.error || "Server error"}${detail}`);
        } else {
          const data = await fallback.json();
          handleChatData(data, question);
        }
        return;
      }

      const err = await res.json().catch(() => ({}));
      const detail = err.detail ? `\n\n${err.detail}` : "";
      removeTyping();
      appendMessage("ai", `Error: ${err.error || "Server error"}${detail}`);
      return;
    }

    removeTyping();
    const streamEl = appendStreamingMessage();
    let answer = "";
    let buffer = "";
    let doneMeta = null;
    let streamError = null;

    const reader = res.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      buffer = parseSseBuffer(buffer, (event, data) => {
        if (event === "done") {
          try { doneMeta = JSON.parse(data); } catch { doneMeta = { answer }; }
          return;
        }
        if (event === "error") {
          try {
            const errObj = JSON.parse(data);
            streamError = new Error(errObj.error || "Stream error");
          } catch {
            streamError = new Error("Stream error");
          }
          return;
        }
        try {
          const payload = JSON.parse(data);
          const delta = payload.delta || "";
          answer += delta;
          streamEl.inner.textContent = answer;
        } catch {
          answer += data;
          streamEl.inner.textContent = answer;
        }
      });
      if (streamError) break;
    }

    if (streamError) throw streamError;
    document.getElementById("streaming-message")?.remove();

    const finalData = doneMeta || { answer };
    if (!finalData.answer) finalData.answer = answer;
    handleChatData(finalData, question);
  } catch (err) {
    removeTyping();
    document.getElementById("streaming-message")?.remove();
    if (err?.name === "AbortError") {
      appendMessage(
        "ai",
        `Request timed out after ${Math.round(REQUEST_TIMEOUT_MS / 1000)}s.\n\nTry again, or reduce output length if the server is busy.`
      );
    } else {
      appendMessage(
        "ai",
        `Could not reach the model server at ${API_BASE}.\n\nMake sure your Flask API server is running and reachable at this URL. If you are using a tunnel, update the API Base URL in Settings.`
      );
    }
  }

  isLoading = false;
  sendBtn.disabled = false;
  chatInput.focus();
}

// ─── RELATED QUESTION CLICK ───────────────────────────────────────────────────
function askRelated(question) {
  setChatQuestion(question, true);
}

// ─── COPY TO CLIPBOARD ────────────────────────────────────────────────────────
async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    showToast("Copied to clipboard!");
  } catch {
    showToast("Could not copy.");
  }
}

function saveAnswer(answer, meta = {}) {
  savedAnswers.unshift({
    question: meta.question || "",
    answer,
    source: meta.source || "",
    ts: Date.now(),
  });
  savedAnswers = savedAnswers.slice(0, 30);
  localStorage.setItem("medai-saved-answers", JSON.stringify(savedAnswers));
  renderSavedAnswers();
  showToast("Answer saved");
}

// ─── CLEAR CHAT ───────────────────────────────────────────────────────────────
function clearChat() {
  chatMessages.innerHTML = `
    <div class="msg msg--ai">
      <div class="msg__avatar">
        ${AI_AVATAR_HTML}
      </div>
      <div class="msg__bubble">
        <div class="msg__bubble-inner"><p>Chat cleared. How can I help you today?</p></div>
      </div>
    </div>`;
  showToast("Chat cleared");
}

// ─── SAVE CONVERSATION (download .txt) ────────────────────────────────────────
function saveConversation() {
  const msgs = [...document.querySelectorAll("#chat-messages .msg")];
  if (msgs.length <= 1) { showToast("Nothing to save yet."); return; }

  const text = msgs.map(m => {
    const role = m.classList.contains("msg--user") ? "You" : "MedAI";
    const inner = m.querySelector(".msg__bubble-inner");
    const content = inner ? inner.innerText.trim() : "";
    return `${role}:\n${content}`;
  }).join("\n\n---\n\n");

  const blob = new Blob([text], { type: "text/plain" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `medai-chat-${Date.now()}.txt`;
  a.click();
  URL.revokeObjectURL(a.href);
  showToast("Conversation saved!");
}

// ─── HISTORY RENDER ───────────────────────────────────────────────────────────
function renderHistory() {
  if (!history.length) {
    historyPanel.innerHTML = `
      <div class="empty-state history-empty">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>
        <strong>No history yet</strong>
        <span>Ask a question to start building your recent topics.</span>
      </div>`;
    return;
  }
  historyPanel.innerHTML = "";
  history.slice(0, 30).forEach(h => {
    const question = String(h.question || "");
    const answer = String(h.answer || "");
    const item = document.createElement("div");
    item.className = "history-item";
    item.innerHTML = `
      <div class="history-item__q">${escapeHtml(question.substring(0, 80))}${question.length > 80 ? "..." : ""}</div>
      <div class="history-item__a">${escapeHtml(answer.substring(0, 130))}${answer.length > 130 ? "..." : ""}</div>`;
    item.addEventListener("click", () => {
      setChatQuestion(question);
    });
    historyPanel.appendChild(item);
  });
}

function renderSavedAnswers() {
  if (!savedPanel) return;
  if (!savedAnswers.length) {
    savedPanel.innerHTML = `
      <div class="empty-state history-empty">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/></svg>
        <strong>No saved answers yet</strong>
        <span>Use the bookmark icon on an answer to save it here.</span>
      </div>`;
    return;
  }

  savedPanel.innerHTML = "";
  savedAnswers.forEach((saved, index) => {
    const question = String(saved.question || "Saved answer");
    const answer = String(saved.answer || "");
    const item = document.createElement("div");
    item.className = "history-item saved-item";
    item.innerHTML = `
      <div class="history-item__q">${escapeHtml(question.substring(0, 100))}${question.length > 100 ? "..." : ""}</div>
      <div class="history-item__a">${escapeHtml(answer.substring(0, 220))}${answer.length > 220 ? "..." : ""}</div>
      <div class="saved-item__actions">
        <button type="button" data-saved-action="copy">Copy</button>
        <button type="button" data-saved-action="remove">Remove</button>
      </div>`;
    item.querySelector('[data-saved-action="copy"]')?.addEventListener("click", (event) => {
      event.stopPropagation();
      copyText(answer);
    });
    item.querySelector('[data-saved-action="remove"]')?.addEventListener("click", (event) => {
      event.stopPropagation();
      savedAnswers.splice(index, 1);
      localStorage.setItem("medai-saved-answers", JSON.stringify(savedAnswers));
      renderSavedAnswers();
      showToast("Saved answer removed");
    });
    item.addEventListener("click", () => {
      if (saved.question) setChatQuestion(saved.question);
    });
    savedPanel.appendChild(item);
  });
}

function escapeHtml(s) {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

// ─── SUMMARIZER ───────────────────────────────────────────────────────────────
async function runSummarize(textareaEl, btnEl, resultEl, resultTextEl) {
  const text = textareaEl.value.trim();
  if (!text) { showToast("Please paste some text first."); return; }
  if (text.length > 5000) { showToast("Text too long (max 5000 chars)."); return; }

  const originalLabel = btnEl.textContent;
  btnEl.disabled = true;
  resultEl.classList.add("visible", "is-loading");
  resultTextEl.textContent = "Summarizing your text...";
  btnEl.textContent = "Summarizing…";
  try {
    const res = await fetchWithTimeout(`${API_BASE}/summarize`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ text }),
    });
    const data = await res.json().catch(() => ({}));

    // If we got a clear summary from the endpoint, use it.
    const summary = data && data.summary ? String(data.summary).trim() : "";
    const isDemo = summary.includes("DEMO MODE") || summary.toLowerCase().includes("demo mode");

    if (res.ok && summary && !isDemo) {
      resultTextEl.textContent = summary;
    } else {
      // Fallback: ask the model via the /chat endpoint (some backends produce better freeform answers)
      // Note: the /chat endpoint enforces a ~1000-char question limit, so truncate safely.
      const prefix = "Please summarize the following medical text in 2–3 sentences, preserving key medical facts:\n\n";
      const maxQ = 900 - prefix.length; // leave room for our instruction
      const shortText = text.length > maxQ ? text.slice(0, maxQ) + "\n\n[truncated]" : text;
      const chatQuestion = prefix + shortText;

      try {
        const chatRes = await fetch(`${API_BASE}/chat`, {
          method:  "POST",
          headers: { "Content-Type": "application/json" },
          body:    JSON.stringify({ question: chatQuestion }),
        });
        const chatData = await chatRes.json().catch(() => ({}));
        const chatAnswer = chatData.answer || chatData.response || "";
        if (chatRes.ok && chatAnswer) {
          resultTextEl.textContent = String(chatAnswer).trim();
        } else {
          resultTextEl.textContent = data.error || chatData.error || "The server could not summarize this text.";
          showToast("Summary request failed.");
        }
      } catch (e) {
        resultTextEl.textContent = `Could not reach the model server at ${API_BASE}.`;
        showToast("Could not reach the server.");
      }
    }
  } catch (err) {
    if (err?.name === "AbortError") {
      resultTextEl.textContent = `Summary request timed out after ${Math.round(REQUEST_TIMEOUT_MS / 1000)}s.`;
      showToast("Summary request timed out.");
    } else {
      resultTextEl.textContent = `Could not reach the model server at ${API_BASE}.`;
      showToast("Could not reach the server.");
    }
  } finally {
    resultEl.classList.remove("is-loading");
    btnEl.disabled = false;
    btnEl.textContent = originalLabel;
  }
}

function attachTextFile(file) {
  if (!file) return;
  if (file.size > 200000) {
    showToast("File too large. Please attach a text file under 200 KB.");
    return;
  }

  const reader = new FileReader();
  reader.onload = () => {
    const text = String(reader.result || "").trim();
    if (!text) {
      showToast("The selected file is empty.");
      return;
    }
    const current = chatInput.value.trim();
    chatInput.value = current ? `${current}\n\n${text}` : text;
    autoResize(chatInput);
    chatInput.focus();
    showToast("Text file attached to your prompt.");
  };
  reader.onerror = () => showToast("Could not read that file.");
  reader.readAsText(file);
}

// ─── TOAST ────────────────────────────────────────────────────────────────────
let toastTimeout;
function showToast(msg) {
  toastEl.textContent = msg;
  toastEl.classList.add("show");
  clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => toastEl.classList.remove("show"), 2500);
}

let typingGlowTimeout;
function showTypingGlow() {
  if (!chatInputShell) return;
  chatInputShell.classList.add("is-typing");
  clearTimeout(typingGlowTimeout);
  typingGlowTimeout = setTimeout(() => {
    chatInputShell.classList.remove("is-typing");
  }, 1200);
}

// ─── EVENT WIRING ─────────────────────────────────────────────────────────────
function bindEvents() {
  // Tab buttons
  document.querySelectorAll(".nav-item[data-tab]").forEach(n => {
    n.addEventListener("click", () => switchTab(n.dataset.tab));
  });

  // Action buttons
  document.querySelectorAll("[data-action]").forEach(b => {
    const action = b.dataset.action;
    b.addEventListener("click", () => {
      if (action === "save")  saveConversation();
      if (action === "clear") clearChat();
    });
  });

  document.querySelectorAll("[data-question]").forEach(b => {
    b.addEventListener("click", () => setChatQuestion(b.dataset.question, true));
  });

  // Chat input
  chatInput.addEventListener("input", () => {
    autoResize(chatInput);
    showTypingGlow();
  });
  chatInput.addEventListener("blur", () => {
    clearTimeout(typingGlowTimeout);
    chatInputShell?.classList.remove("is-typing");
  });
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  sendBtn.addEventListener("click", sendMessage);

  attachBtn?.addEventListener("click", () => attachInput?.click());
  attachInput?.addEventListener("change", () => {
    attachTextFile(attachInput.files?.[0]);
    attachInput.value = "";
  });

  // Summarizers
  sumBtn.addEventListener("click", () => runSummarize(sumInput, sumBtn, sumResult, sumResultText));
  rightSumBtn.addEventListener("click", () => runSummarize(rightSumInput, rightSumBtn, rightSumResult, rightSumText));

  // Drop zone -> focus textarea
  document.querySelector(".summarizer-drop")?.addEventListener("click", () => sumInput.focus());

const clearHistoryBtn = document.getElementById("clear-history-btn");
const exportChatBtn = document.getElementById("export-chat-btn");
const exportSettingsBtn = document.getElementById("export-settings-btn");

// Clear history
clearHistoryBtn?.addEventListener("click", () => {
  history = [];
  localStorage.removeItem("medai-history");
  renderHistory();
  showToast("History cleared");
});
exportChatBtn?.addEventListener("click", saveConversation);
exportSettingsBtn?.addEventListener("click", saveConversation);

// SETTINGS LOGIC

const themeSelect = document.getElementById("theme-select");
const glassSlider = document.getElementById("glass-slider");
const showConfidence = document.getElementById("show-confidence");
const apiBaseInput = document.getElementById("api-base-input");

// LOAD SETTINGS
const settings = JSON.parse(localStorage.getItem("medai-settings") || "{}");

function applyTheme(value) {
  document.body.classList.toggle("light", value === "light");
}

function applyGlass(value) {
  const glass = Number(value) || 18;
  const strength = (glass - 5) / 35;
  const tint = 0.06 + strength * 0.16;
  const darkTint = 0.04 + strength * 0.10;
  const edge = 0.12 + strength * 0.24;

  [document.documentElement, document.body].forEach((target) => {
    target.style.setProperty("--blur", `${glass}px`);
    target.style.setProperty("--lg-tint", `rgba(255, 255, 255, ${tint.toFixed(3)})`);
    target.style.setProperty("--lg-tint-dark", `rgba(255, 255, 255, ${darkTint.toFixed(3)})`);
    target.style.setProperty("--lg-edge", `rgba(255, 255, 255, ${edge.toFixed(3)})`);
  });
}

function saveSettings() {
  persistSettings({
    theme: themeSelect.value,
    glass: glassSlider.value,
    showConfidence: showConfidence.checked,
    apiBase: API_BASE
  });
}

themeSelect.value = settings.theme || "dark";
glassSlider.value = settings.glass || 18;
apiBaseInput.value = API_BASE;

const savedShowConfidence = settings.showConfidence !== false;
showConfidence.checked = savedShowConfidence;

applyTheme(themeSelect.value);
applyGlass(glassSlider.value);
document.body.classList.toggle("hide-confidence", !savedShowConfidence);

themeSelect.addEventListener("change", () => {
  applyTheme(themeSelect.value);
  saveSettings();
});

glassSlider.addEventListener("input", () => {
  applyGlass(glassSlider.value);
  saveSettings();
});

apiBaseInput.addEventListener("change", () => {
  const nextBase = apiBaseInput.value.trim().replace(/\/*$/, "");
  if (!nextBase) {
    apiBaseInput.value = API_BASE;
    return;
  }
  API_BASE = nextBase;
  saveSettings();
  checkHealth();
  showToast("API URL updated");
});

// CHECKBOXES
showConfidence.addEventListener("change", () => {
  document.body.classList.toggle("hide-confidence", !showConfidence.checked);
  saveSettings();
});

// RESET SETTINGS
document.getElementById("reset-settings-btn").addEventListener("click", () => {
  localStorage.removeItem("medai-settings");
  location.reload();
});
}


// ─── INIT ─────────────────────────────────────────────────────────────────────
function init() {
  bindEvents();
  renderSavedAnswers();
  checkHealth();
  setInterval(checkHealth, 15000);
  chatInput.focus();
}

document.addEventListener("DOMContentLoaded", init);
