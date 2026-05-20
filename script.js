/* ═══════════════════════════════════════════════════════════════════════
  MedAI — Chatbot Frontend Logic
  ═══════════════════════════════════════════════════════════════════════ */

const DEFAULT_API = "http://localhost:8000";
// Load persisted settings early so API_BASE can be configured
const _persistedSettings = JSON.parse(localStorage.getItem("medai-settings") || "{}");
let API_BASE = _persistedSettings.apiBase || DEFAULT_API;

// ─── State ────────────────────────────────────────────────────────────────────
let isLoading = false;
let history   = JSON.parse(localStorage.getItem("medai-history") || "[]");

// ─── DOM Refs ─────────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

const chatMessages   = $("chat-messages");
const chatInput      = $("chat-input");
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
const apiBaseInput   = $("api-base-input");
const historyPanel   = $("history-panel");
const toastEl        = $("toast");

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

      // Try common safe probes
      let r = await probe("GET", "/health");
      if (r) return r;
      r = await probe("GET", "/predict/health");
      if (r) return r;
      r = await probe("GET", ""); // root
      if (r) return r;
      r = await probe("GET", "/docs");
      if (r) return r;
      r = await probe("HEAD", "");
      if (r) return r;
      // OPTIONS on likely chat endpoints (CORS preflight may respond)
      r = await probe("OPTIONS", "/chat");
      if (r) return r;
      r = await probe("OPTIONS", "/predict/chat");
      if (r) return r;
      // No success
      return null;
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
    if (apiBaseInput) apiBaseInput.value = API_BASE;
    saveSettings();
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
    av.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>
      </svg>`;
    wrap.appendChild(av);
  }

  const bubble = document.createElement("div");
  bubble.className = "msg__bubble";

  // Inner content holder
  const inner = document.createElement("div");
  inner.className = "msg__bubble-inner";
  inner.innerHTML = formatBody(content);
  bubble.appendChild(inner);

  // Meta (confidence + source + copy)
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
        <span><strong>Confidence:</strong> ${meta.confidence}%</span>
      </span>
      <span class="msg__source">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2zM22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>
        <span style="color:var(--lg-blue-bright)">${meta.source}</span>
      </span>
      <button class="msg__copy" title="Copy">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
      </button>`;
    metaDiv.querySelector(".msg__copy").addEventListener("click", () => copyText(content));
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
}

// ─── TYPING INDICATOR ─────────────────────────────────────────────────────────
function showTyping() {
  const div = document.createElement("div");
  div.className = "msg msg--ai";
  div.id = "typing-indicator";
  div.innerHTML = `
    <div class="msg__avatar">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>
      </svg>
    </div>
    <div class="msg__bubble">
      <div class="msg__bubble-inner typing-bubble">
        <div class="typing-dots"><span></span><span></span><span></span></div>
      </div>
    </div>`;
  chatMessages.appendChild(div);
}
function removeTyping() {
  document.getElementById("typing-indicator")?.remove();
}

// ─── SEND MESSAGE ─────────────────────────────────────────────────────────────
async function sendMessage() {
  const question = chatInput.value.trim();
  if (!question || isLoading) return;

  isLoading = true;
  sendBtn.disabled = true;
  chatInput.value = "";
  chatInput.style.height = "auto";

  appendMessage("user", question);
  showTyping();

  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ question }),
    });
    removeTyping();

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      appendMessage("ai", `⚠️ Error: ${err.error || "Server error"}`);
    } else {
      const data = await res.json();

      // Support multiple backend shapes:
      // - Local Flask: { answer, confidence, source, related_questions }
      // - Kaggle FastAPI notebook: { response }
      const answerText = data.answer || data.response || data.answer_text || data.summary || JSON.stringify(data);
      const confidence = data.confidence || data.confidence_score || null;
      const source     = data.source || data.from || "Remote API";
      const related    = data.related_questions || data.related || [];

      appendMessage("ai", answerText, {
        confidence: confidence ?? 0,
        source:     source,
        related:    related,
      });

      // Save to history
      history.unshift({ question, answer: data.answer, ts: Date.now() });
      if (history.length > 50) history.pop();
      localStorage.setItem("medai-history", JSON.stringify(history));
    }
  } catch {
    removeTyping();
    appendMessage(
      "ai",
      `⚠️ Could not reach the model server at ${API_BASE}.\n\nMake sure your FastAPI server is running and reachable at this URL (check network, CORS, and that the host supports HTTPS).`
    );
  }

  isLoading = false;
  sendBtn.disabled = false;
  chatInput.focus();
}

// ─── RELATED QUESTION CLICK ───────────────────────────────────────────────────
function askRelated(question) {
  chatInput.value = question;
  sendMessage();
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

// ─── CLEAR CHAT ───────────────────────────────────────────────────────────────
function clearChat() {
  chatMessages.innerHTML = `
    <div class="msg msg--ai">
      <div class="msg__avatar">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>
        </svg>
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
    historyPanel.innerHTML = `<div class="history-empty">No history yet. Ask a question to get started!</div>`;
    return;
  }
  historyPanel.innerHTML = "";
  history.slice(0, 30).forEach(h => {
    const item = document.createElement("div");
    item.className = "history-item";
    item.innerHTML = `
      <div class="history-item__q">${escapeHtml(h.question.substring(0, 80))}${h.question.length > 80 ? "…" : ""}</div>
      <div class="history-item__a">${escapeHtml(h.answer.substring(0, 130))}…</div>`;
    item.addEventListener("click", () => {
      switchTab("chat");
      chatInput.value = h.question;
      chatInput.focus();
    });
    historyPanel.appendChild(item);
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
  btnEl.textContent = "Summarizing…";

  try {
    const res = await fetch(`${API_BASE}/summarize`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ text }),
    });
    const data = await res.json();
    resultTextEl.textContent = data.summary || data.error || "No summary returned.";
    resultEl.classList.add("visible");
  } catch {
    showToast("Could not reach the server.");
  }
  btnEl.disabled = false;
  btnEl.textContent = originalLabel;
}

// ─── TOAST ────────────────────────────────────────────────────────────────────
let toastTimeout;
function showToast(msg) {
  toastEl.textContent = msg;
  toastEl.classList.add("show");
  clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => toastEl.classList.remove("show"), 2500);
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

  // Chat input
  chatInput.addEventListener("input", () => autoResize(chatInput));
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  sendBtn.addEventListener("click", sendMessage);

  // Summarizers
  sumBtn.addEventListener("click", () => runSummarize(sumInput, sumBtn, sumResult, sumResultText));
  rightSumBtn.addEventListener("click", () => runSummarize(rightSumInput, rightSumBtn, rightSumResult, rightSumText));

  // API Base input (network settings)
  if (apiBaseInput) {
    apiBaseInput.value = API_BASE || DEFAULT_API;
    apiBaseInput.addEventListener("change", () => {
      const v = apiBaseInput.value.trim();
      if (v) API_BASE = v;
      saveSettings();
      checkHealth();
      showToast("API URL updated");
    });
  }

  // Drop zone -> focus textarea
  document.querySelector(".summarizer-drop")?.addEventListener("click", () => sumInput.focus());

const clearHistoryBtn = document.getElementById("clear-history-btn");

// Clear history
clearHistoryBtn?.addEventListener("click", () => {
  history = [];
  localStorage.removeItem("medai-history");
  renderHistory();
  showToast("History cleared");
});

// SETTINGS LOGIC

const themeSelect = document.getElementById("theme-select");
const glassSlider = document.getElementById("glass-slider");
const showConfidence = document.getElementById("show-confidence");

// LOAD SETTINGS
const settings = _persistedSettings || {};

// Apply theme properly
const savedTheme = settings.theme || "dark";

// Set dropdown FIRST
themeSelect.value = savedTheme;

themeSelect.addEventListener("change", () => {
  const value = themeSelect.value;

  document.body.classList.toggle("light", value === "light");

  saveSettings();
});

// Then apply class
if (savedTheme === "light") {
  document.body.classList.add("light");
} else {
  document.body.classList.remove("light");
}

const savedBlur = settings.glass || 10;

glassSlider.value = savedBlur;
document.documentElement.style.setProperty("--blur", savedBlur + "px");

// SAVE FUNCTION
function saveSettings() {
  const newSettings = {
    theme: themeSelect.value,
    glass: glassSlider.value,
    apiBase: (apiBaseInput && apiBaseInput.value.trim()) || API_BASE,
    showConfidence: showConfidence.checked
  };
  localStorage.setItem("medai-settings", JSON.stringify(newSettings));
}

// THEME CHANGE
themeSelect.addEventListener("change", () => {
  const value = themeSelect.value;

  if (value === "light") {
    document.body.classList.add("light");
  } else {
    document.body.classList.remove("light"); // default = dark
  }

  saveSettings();
});

// GLASS CONTROL
glassSlider.addEventListener("input", () => {
  document.documentElement.style.setProperty("--blur", glassSlider.value + "px");
  saveSettings();
});

// CHECKBOXES
showConfidence.addEventListener("change", saveSettings);

// RESET SETTINGS
document.getElementById("reset-settings-btn").addEventListener("click", () => {
  localStorage.removeItem("medai-settings");
  location.reload();
});
}


// ─── INIT ─────────────────────────────────────────────────────────────────────
function init() {
  bindEvents();
  checkHealth();
  setInterval(checkHealth, 15000);
  chatInput.focus();
}

document.addEventListener("DOMContentLoaded", init);
