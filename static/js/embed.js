(async function() {
  const scriptEl = document.currentScript || document.querySelector('script[data-api-key]');

  function getApiKeyFromScript(el) {
    if (!el) return null;
    const fromAttr = el.getAttribute("data-api-key");
    if (fromAttr) return fromAttr;
    if (el.src) {
      try {
        const url = new URL(el.src, window.location.href);
        const fromQuery = url.searchParams.get("api_key");
        if (fromQuery) return fromQuery;
      } catch (e) {}
    }
    return null;
  }

  const API_KEY = getApiKeyFromScript(scriptEl);
  const isDemo = !!(window.__THEOCHAT_DEMO__ && window.__THEOCHAT_DEMO__.page === "marketing");
  if (!API_KEY) {
    console.error("[TheoChat] API key missing. Add data-api-key to the script tag or ?api_key=... in the src.");
    return;
  }
  const baseUrl =
    (scriptEl && scriptEl.getAttribute("data-base-url")) ||
    (scriptEl ? new URL(scriptEl.src, window.location.href).origin : "");
  const globalBase = window.__CHATBOT_BASE_URL;
  const BASE_URL = (baseUrl || globalBase);
  if (!BASE_URL) {
    console.error("[TheoChat] BASE_URL not set. Add data-base-url to the script tag or set window.__CHATBOT_BASE_URL.");
    return;
  }
  const demoSuffix = isDemo ? "&demo=1" : "";
  const ENDPOINT = BASE_URL.replace(/\/$/, "") + "/chat_stream?api_key=" + encodeURIComponent(API_KEY) + demoSuffix;
  const CONFIG_URL = BASE_URL.replace(/\/$/, "") + "/widget_config?api_key=" + encodeURIComponent(API_KEY) + demoSuffix;
  const logoUrl = baseUrl
    ? `${baseUrl.replace(/\/+$/, "")}/static/img/theochat-logo-mark.png`
    : "/static/img/theochat-logo-mark.png";

  const DEFAULTS = {
    primary: "#0f766e",
    secondary: "#22c55e",
    font: "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif",
    radius: "12px",
  };

  function readCSSVar(names = []) {
    const style = getComputedStyle(document.documentElement);
    for (const name of names) {
      const val = style.getPropertyValue(name);
      if (val && val.trim()) return val.trim();
    }
    return null;
  }

  const inferredPrimary = readCSSVar(["--primary", "--color-primary", "--brand", "--accent", "--theme-primary"]);
  const inferredSecondary = readCSSVar(["--secondary", "--color-secondary", "--brand-secondary", "--accent-secondary"]);

  let theme = { ...DEFAULTS };
  try {
    const res = await fetch(CONFIG_URL);
    if (res.ok) {
      const data = await res.json();
      theme = {
        primary: data.theme_primary_color || DEFAULTS.primary,
        secondary: data.theme_secondary_color || DEFAULTS.secondary,
        font: data.theme_font_family || DEFAULTS.font,
        radius: data.theme_border_radius || DEFAULTS.radius,
      };
    } else {
      theme = {
        primary: inferredPrimary || DEFAULTS.primary,
        secondary: inferredSecondary || DEFAULTS.secondary,
        font: DEFAULTS.font,
        radius: DEFAULTS.radius,
      };
    }
  } catch (e) {
    theme = {
      primary: inferredPrimary || DEFAULTS.primary,
      secondary: inferredSecondary || DEFAULTS.secondary,
      font: DEFAULTS.font,
      radius: DEFAULTS.radius,
    };
  }

  const root = document.documentElement;
  root.style.setProperty("--theochat-primary", theme.primary);
  root.style.setProperty("--theochat-secondary", theme.secondary);
  root.style.setProperty("--theochat-font-family", theme.font);
  root.style.setProperty("--theochat-radius", theme.radius);

  function hexToRgb(hex) {
    const cleaned = hex.replace("#", "");
    if (cleaned.length === 3) {
      const r = cleaned[0] + cleaned[0];
      const g = cleaned[1] + cleaned[1];
      const b = cleaned[2] + cleaned[2];
      return [parseInt(r, 16), parseInt(g, 16), parseInt(b, 16)];
    }
    if (cleaned.length === 6) {
      return [parseInt(cleaned.slice(0, 2), 16), parseInt(cleaned.slice(2, 4), 16), parseInt(cleaned.slice(4, 6), 16)];
    }
    return null;
  }

  function luminance(hex) {
    const rgb = hexToRgb(hex);
    if (!rgb) return 0.5;
    const [r, g, b] = rgb.map((v) => {
      const n = v / 255;
      return n <= 0.03928 ? n / 12.92 : Math.pow((n + 0.055) / 1.055, 2.4);
    });
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  }

  const primaryLum = luminance(theme.primary.toLowerCase());
  const launcherText = primaryLum > 0.62 ? "#0b1224" : "#f8fafc";

  function normalizeRadius(val) {
    if (!val) return "999px";
    const match = String(val).match(/([\d\.]+)px/);
    if (!match) return val;
    const num = parseFloat(match[1]);
    const clamped = Math.max(num, 14);
    return `${clamped}px`;
  }

  const launcherRadius = normalizeRadius(theme.radius ? theme.radius : "999px");
  function normalizePanelRadius(val) {
    const fallback = DEFAULTS.radius;
    if (!val) return fallback;
    const match = String(val).match(/([\d\.]+)px/);
    if (!match) return fallback;
    const num = parseFloat(match[1]);
    const clamped = Math.min(Math.max(num, 12), 20);
    return `${clamped}px`;
  }
  const panelRadius = normalizePanelRadius(theme.radius);
  root.style.setProperty("--tc-primary", theme.primary);
  root.style.setProperty("--tc-secondary", theme.secondary);
  root.style.setProperty("--tc-font", theme.font);
  root.style.setProperty("--tc-radius-launcher", launcherRadius);
  root.style.setProperty("--tc-launcher-text", launcherText);

  const launcherBottom = isDemo ? 24 : 16;
  const launcherRight = isDemo ? 24 : 16;
  const launcherHeight = 52; // approximate rendered height with padding/icon/label
  const gapBetween = 12;
  const hoverClearance = 12;
  const panelBottomVal = launcherBottom + launcherHeight + gapBetween + hoverClearance;

  root.style.setProperty("--tc-launcher-bottom", `${launcherBottom}px`);
  root.style.setProperty("--tc-launcher-right", `${launcherRight}px`);
  root.style.setProperty("--tc-panel-bottom", `${panelBottomVal}px`);
  root.style.setProperty("--tc-panel-right", `${launcherRight}px`);

  const styleEl = document.createElement("style");
  styleEl.textContent = `
    .theochat-launcher {
      position: fixed;
      bottom: var(--tc-launcher-bottom, ${launcherBottom}px);
      right: var(--tc-launcher-right, ${launcherRight}px);
      z-index: 9999;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 12px 16px;
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: var(--tc-radius-launcher, 999px);
      background: linear-gradient(135deg, var(--tc-primary, #0f766e), var(--tc-secondary, #22c55e));
      color: var(--tc-launcher-text, #f8fafc);
      font-family: var(--tc-font, ${DEFAULTS.font});
      font-weight: 600;
      font-size: 14px;
      box-shadow: 0 10px 24px rgba(0,0,0,0.22), 0 4px 12px rgba(0,0,0,0.14);
      cursor: pointer;
      outline: none;
      transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
    }
    .theochat-powered {
      display: block;
      margin: 10px 0 0;
      font-size: 12px;
      text-align: center;
      color: rgba(15,23,42,0.7);
      text-decoration: none;
      opacity: 0.8;
    }
    .theochat-powered:hover {
      opacity: 1;
      text-decoration: underline;
    }
    .dark .theochat-powered {
      color: rgba(255,255,255,0.78);
    }
    .theochat-launcher::before {
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(120deg, rgba(255,255,255,0.2), transparent 40%);
      opacity: 0;
      transition: opacity 0.2s ease;
      border-radius: inherit;
      pointer-events: none;
    }
    .theochat-launcher:hover::before {
      opacity: 0.6;
    }
    .theochat-launcher:hover {
      transform: translateY(-2px) scale(1.02);
      box-shadow: 0 14px 28px rgba(0,0,0,0.28), 0 6px 14px rgba(0,0,0,0.18);
    }
    .theochat-launcher:active {
      transform: translateY(0) scale(0.99);
      box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    .theochat-launcher:focus-visible {
      box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.35), 0 10px 24px rgba(0,0,0,0.22);
    }
    .theochat-launcher__icon {
      width: 18px;
      height: 18px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }
    .theochat-launcher__icon svg {
      width: 18px;
      height: 18px;
      display: block;
      fill: currentColor;
    }
    @media (prefers-reduced-motion: reduce) {
      .theochat-launcher,
      .theochat-launcher::before {
        transition: none !important;
      }
      .theochat-launcher:hover {
        transform: none;
      }
      .theochat-launcher:active {
        transform: none;
      }
    }
    @media (max-width: 480px) {
      .theochat-launcher {
        padding: 10px 14px;
        bottom: ${isDemo ? "20px" : "14px"};
        right: ${isDemo ? "20px" : "14px"};
      }
    }
  `;
  document.head.appendChild(styleEl);

  const btn = document.createElement("button");
  const defaultLabel = "Chat with us";
  const customLabel = scriptEl && scriptEl.getAttribute("data-button-label");
  btn.className = "theochat-launcher";
  btn.type = "button";
  btn.id = "theochat-launcher";
  btn.innerHTML = `
    <span class="theochat-launcher__icon" aria-hidden="true">
      <svg viewBox="0 0 24 24" focusable="false">
        <path d="M4 4.75C4 3.784 4.784 3 5.75 3h12.5C19.216 3 20 3.784 20 4.75v8.5c0 .966-.784 1.75-1.75 1.75H8.414l-3.02 2.817A.75.75 0 0 1 4 17.25v-12.5Z"></path>
      </svg>
    </span>
    <span class="theochat-launcher__label">${customLabel || defaultLabel}</span>
  `;

  const panel = document.createElement("div");
  panel.style.position = "fixed";
  panel.id = "theochat-panel";
  // Panel styles must not share launcher radius rules.
  panel.style.bottom = "var(--tc-panel-bottom)";
  panel.style.right = "var(--tc-panel-right)";
  panel.style.width = "320px";
  panel.style.height = "470px";
  panel.style.background = "#fff";
  panel.style.color = "#111";
  panel.style.border = "1px solid #e5e7eb";
  // Panel styles must not share launcher radius rules.
  panel.style.borderRadius = panelRadius;
  panel.style.boxShadow = "0 10px 30px rgba(0,0,0,0.25)";
  panel.style.display = "none";
  panel.style.flexDirection = "column";
  panel.style.overflow = "hidden";
  const WIDGET_FONT = "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Inter, Arial, sans-serif";
  panel.style.fontFamily = isDemo ? "var(--theochat-font-family, " + DEFAULTS.font + ")" : WIDGET_FONT;
  panel.style.opacity = "0";
  panel.style.transform = "translateY(8px)";
  panel.style.transition = "opacity 0.2s ease, transform 0.2s ease";

  const resizer = document.createElement("div");
  resizer.style.position = "absolute";
  resizer.style.left = "4px";
  resizer.style.top = "4px";
  resizer.style.width = "16px";
  resizer.style.height = "16px";
  resizer.style.cursor = "nwse-resize";
  resizer.style.background = "transparent";
  resizer.style.zIndex = "10000";

  let isResizing = false;
  let startX, startY, startW, startH;

  resizer.addEventListener("mousedown", (e) => {
    isResizing = true;
    startX = e.clientX;
    startY = e.clientY;
    startW = panel.offsetWidth;
    startH = panel.offsetHeight;
    e.preventDefault();
    e.stopPropagation();
  });

  window.addEventListener("mousemove", (e) => {
    if (!isResizing) return;
    const dx = startX - e.clientX;
    const dy = startY - e.clientY;
    const minW = 220;
    const maxW = 600;
    const minH = 280;
    const maxH = 800;
    const newW = Math.min(maxW, Math.max(minW, startW + dx));
    const newH = Math.min(maxH, Math.max(minH, startH + dy));
    panel.style.width = newW + "px";
    panel.style.height = newH + "px";
  });

  window.addEventListener("mouseup", () => {
    isResizing = false;
  });

  function openPanel() {
    panel.style.display = "flex";
    requestAnimationFrame(() => {
      panel.style.opacity = "1";
      panel.style.transform = "translateY(0)";
    });
    hideDemoHelper();
    try {
      window.dispatchEvent(new CustomEvent("theochat:open"));
    } catch (e) {}
  }

  function closePanel() {
    panel.style.opacity = "0";
    panel.style.transform = "translateY(8px)";
    setTimeout(() => {
      panel.style.display = "none";
    }, 200);
    showDemoHelper();
    try {
      window.dispatchEvent(new CustomEvent("theochat:close"));
    } catch (e) {}
  }

  const header = document.createElement("div");
  header.style.display = "flex";
  header.style.alignItems = "center";
  header.style.gap = "8px";
  header.style.fontWeight = "600";
  header.style.padding = "10px 12px";
  header.style.borderBottom = "1px solid #e5e7eb";
  const headerAvatar = document.createElement("img");
  headerAvatar.src = logoUrl;
  headerAvatar.alt = "TheoChat";
  headerAvatar.style.width = "20px";
  headerAvatar.style.height = "20px";
  headerAvatar.style.borderRadius = "999px";
  headerAvatar.style.objectFit = "cover";
  const headerTitle = document.createElement("span");
  headerTitle.textContent = "TheoChat";
  header.appendChild(headerAvatar);
  header.appendChild(headerTitle);

  const chatArea = document.createElement("div");
  chatArea.style.flex = "1";
  chatArea.style.padding = "10px";
  chatArea.style.overflowY = "auto";
  chatArea.style.display = "flex";
  chatArea.style.flexDirection = "column";
  chatArea.style.gap = "8px";
  chatArea.style.fontSize = "14px";
  chatArea.style.background = "#f9fafb";

  function addBubble(text, isUser) {
    const div = document.createElement("div");
    div.textContent = text;
    div.style.padding = "8px 10px";
    div.style.borderRadius = "10px";
    div.style.maxWidth = "80%";
    div.style.whiteSpace = "pre-wrap";
    div.style.alignSelf = isUser ? "flex-end" : "flex-start";
    if (isUser) {
      div.style.background = "var(--theochat-primary, #2563eb)";
      div.style.color = "#fff";
    } else {
      div.style.background = "#e5e7eb";
      div.style.color = "#111827";
    }
    chatArea.appendChild(div);
    chatArea.scrollTop = chatArea.scrollHeight;
    return div;
  }

  addBubble("Hi, I'm Theo, your website's AI assistant from TheoChat. Ask me anything about this business.", false);

  const inputRow = document.createElement("div");
  inputRow.style.display = "flex";
  inputRow.style.alignItems = "center";
  inputRow.style.gap = "6px";
  inputRow.style.padding = "10px";
  inputRow.style.borderTop = "1px solid #e5e7eb";
  const input = document.createElement("input");
  input.type = "text";
  input.placeholder = "Ask Theo a question about this site...";
  input.style.flex = "1 1 auto";
  input.style.minWidth = "0";
  input.style.fontSize = "14px";
  input.style.padding = "8px";
  input.style.border = "1px solid #d1d5db";
  input.style.borderRadius = "var(--theochat-radius, 8px)";
  input.style.fontFamily = isDemo ? "var(--theochat-font-family, " + DEFAULTS.font + ")" : WIDGET_FONT;
  const sendBtn = document.createElement("button");
  sendBtn.textContent = "Send";
  sendBtn.style.padding = "8px 12px";
  sendBtn.style.border = "none";
  sendBtn.style.borderRadius = "var(--theochat-radius, 8px)";
  sendBtn.style.background = "var(--theochat-primary, #2563eb)";
  sendBtn.style.color = "#fff";
  sendBtn.style.cursor = "pointer";
  sendBtn.style.fontFamily = isDemo ? "var(--theochat-font-family, " + DEFAULTS.font + ")" : WIDGET_FONT;
  sendBtn.style.flex = "0 0 auto";
  sendBtn.style.whiteSpace = "nowrap";
  sendBtn.style.minWidth = "68px";

  inputRow.appendChild(input);
  inputRow.appendChild(sendBtn);

  const contactLink = document.createElement("button");
  contactLink.textContent = "Contact us";
  contactLink.style.fontSize = "12px";
  contactLink.style.background = "transparent";
  contactLink.style.border = "none";
  contactLink.style.color = "#4b5563";
  contactLink.style.cursor = "pointer";
  contactLink.style.padding = "0 10px 8px";
  contactLink.style.alignSelf = "flex-start";

  const contactForm = document.createElement("div");
  contactForm.style.display = "none";
  contactForm.style.flexDirection = "column";
  contactForm.style.gap = "6px";
  contactForm.style.padding = "10px";
  contactForm.style.borderTop = "1px solid #e5e7eb";

  const nameInput = document.createElement("input");
  nameInput.type = "text";
  nameInput.placeholder = "Your name (optional)";
  nameInput.style.padding = "8px";
  nameInput.style.border = "1px solid #d1d5db";
  nameInput.style.borderRadius = "var(--theochat-radius, 8px)";
  const emailInput = document.createElement("input");
  emailInput.type = "email";
  emailInput.placeholder = "Your email";
  emailInput.style.padding = "8px";
  emailInput.style.border = "1px solid #d1d5db";
  emailInput.style.borderRadius = "var(--theochat-radius, 8px)";
  const phoneInput = document.createElement("input");
  phoneInput.type = "text";
  phoneInput.placeholder = "Your phone (optional)";
  phoneInput.style.padding = "8px";
  phoneInput.style.border = "1px solid #d1d5db";
  phoneInput.style.borderRadius = "var(--theochat-radius, 8px)";
  const messageInput = document.createElement("textarea");
  messageInput.placeholder = "Your message";
  messageInput.rows = 3;
  messageInput.style.padding = "8px";
  messageInput.style.border = "1px solid #d1d5db";
  messageInput.style.borderRadius = "var(--theochat-radius, 8px)";
  messageInput.style.resize = "vertical";
  const contactSend = document.createElement("button");
  contactSend.textContent = "Send";
  contactSend.style.padding = "8px 12px";
  contactSend.style.border = "none";
  contactSend.style.borderRadius = "var(--theochat-radius, 8px)";
  contactSend.style.background = "var(--theochat-primary, #2563eb)";
  contactSend.style.color = "#fff";
  contactSend.style.cursor = "pointer";
  contactSend.style.fontFamily = "var(--theochat-font-family, " + DEFAULTS.font + ")";
  const contactCancel = document.createElement("button");
  contactCancel.textContent = "Cancel";
  contactCancel.style.padding = "8px 12px";
  contactCancel.style.border = "none";
  contactCancel.style.borderRadius = "var(--theochat-radius, 8px)";
  contactCancel.style.background = "#e5e7eb";
  contactCancel.style.color = "#111827";
  contactCancel.style.cursor = "pointer";

  const contactStatus = document.createElement("div");
  contactStatus.style.fontSize = "12px";
  contactStatus.style.color = "#4b5563";

  contactForm.appendChild(nameInput);
  contactForm.appendChild(emailInput);
  contactForm.appendChild(phoneInput);
  contactForm.appendChild(messageInput);
  contactForm.appendChild(contactSend);
  contactForm.appendChild(contactCancel);
  contactForm.appendChild(contactStatus);

  const contactWrapper = document.createElement("div");
  contactWrapper.appendChild(contactLink);
  contactWrapper.appendChild(contactForm);
  contactWrapper.style.background = "#f9fafb";
  contactWrapper.style.paddingBottom = "10px";
  contactWrapper.style.borderTop = "1px solid #e5e7eb";
  contactWrapper.style.paddingLeft = "10px";
  contactWrapper.style.paddingRight = "10px";

  const powered = document.createElement("a");
  const marketingUrl = (scriptEl && scriptEl.getAttribute("data-theochat-site")) || "https://www.theochat.co.uk/";
  powered.href = marketingUrl;
  powered.target = "_blank";
  powered.rel = "noopener noreferrer";
  powered.textContent = "Powered by TheoChat";
  powered.className = "theochat-powered";
  powered.style.display = "block";
  powered.style.margin = "8px 0 0";
  powered.style.fontSize = "12px";
  powered.style.opacity = "0.8";
  powered.style.textDecoration = "none";
  powered.style.color = "#0f172a";
  powered.style.textAlign = "center";
  powered.addEventListener("mouseover", () => { powered.style.opacity = "1"; powered.style.textDecoration = "underline"; });
  powered.addEventListener("mouseout", () => { powered.style.opacity = "0.8"; powered.style.textDecoration = "none"; });

  const container = document.createElement("div");
  container.style.display = "flex";
  container.style.flexDirection = "column";
  container.style.height = "100%";
  container.appendChild(header);
  container.appendChild(chatArea);
  container.appendChild(inputRow);
  container.appendChild(contactWrapper);
  contactWrapper.appendChild(powered);

  panel.appendChild(container);
  panel.appendChild(resizer);

  let demoHelper = null;

  function hideDemoHelper() {
    if (!demoHelper) return;
    demoHelper.classList.remove("is-visible");
  }

  function showDemoHelper() {
    if (!demoHelper) return;
    demoHelper.classList.add("is-visible");
    adjustHelperPosition();
  }

  function adjustHelperPosition() {
    if (!demoHelper || !btn) return;
    demoHelper.style.bottom = "110px";
    const hRect = demoHelper.getBoundingClientRect();
    const bRect = btn.getBoundingClientRect();
    const overlap = !(hRect.right < bRect.left || hRect.left > bRect.right || hRect.bottom < bRect.top || hRect.top > bRect.bottom);
    if (overlap) {
      const extra = bRect.height + 20;
      demoHelper.style.bottom = `${110 + extra}px`;
    }
  }

  function createDemoHelper() {
    demoHelper = document.createElement("div");
    demoHelper.className = "demo-widget-helper";
    demoHelper.id = "tc-demo-helper";
    demoHelper.innerHTML = `
      <div class="demo-widget-helper__title">This is what the TheoChat chatbot would look like on your page.</div>
      <p>Try asking a question, or click Contact us.</p>
    `;
    document.body.appendChild(demoHelper);
    setTimeout(() => {
      showDemoHelper();
    }, 250);
    let resizeTimer = null;
    window.addEventListener("resize", () => {
      if (!demoHelper) return;
      if (resizeTimer) clearTimeout(resizeTimer);
      resizeTimer = setTimeout(adjustHelperPosition, 120);
    });
  }

  btn.addEventListener("click", () => {
    const isHidden = panel.style.display === "none";
    if (isHidden) {
      openPanel();
    } else {
      closePanel();
    }
  });

  if (isDemo) {
    createDemoHelper();
  }

  async function sendMessage() {
    const text = input.value.trim();
    if (!text) return;
    addBubble(text, true);
    input.value = "";
    const botDiv = addBubble("...", false);
    sendBtn.disabled = true;
    input.disabled = true;
    try {
      const res = await fetch(ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });
      if (!res.ok || !res.body) {
        botDiv.textContent = "Error: " + res.status;
        return;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let fullText = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        fullText += chunk;
        botDiv.textContent = fullText;
        chatArea.scrollTop = chatArea.scrollHeight;
      }
    } catch (e) {
      botDiv.textContent = "Network error";
    } finally {
      sendBtn.disabled = false;
      input.disabled = false;
      input.focus();
    }
  }

  sendBtn.addEventListener("click", sendMessage);
  input.addEventListener("keypress", (e) => { if (e.key === "Enter") sendMessage(); });

  function toggleContactForm(show) {
    contactForm.style.display = show ? "flex" : "none";
    contactStatus.textContent = "";
  }

  contactLink.addEventListener("click", () => {
    const shouldShow = contactForm.style.display === "none";
    toggleContactForm(shouldShow);
    if (shouldShow) {
      panel.scrollTop = panel.scrollHeight;
    }
  });

  contactCancel.addEventListener("click", (e) => {
    e.preventDefault();
    toggleContactForm(false);
  });

  contactSend.addEventListener("click", async () => {
    const name = nameInput.value.trim();
    const email = emailInput.value.trim();
    const phone = phoneInput.value.trim();
    const message = messageInput.value.trim();
    if (!email) {
      contactStatus.textContent = "Please enter your email so the business can reply.";
      contactStatus.style.color = "#b91c1c";
      return;
    }
    if (!message) {
      contactStatus.textContent = "Please enter a message.";
      contactStatus.style.color = "#b91c1c";
      return;
    }
    contactSend.disabled = true;
    contactStatus.textContent = "Sending...";
    contactStatus.style.color = "#4b5563";
    try {
      const res = await fetch(BASE_URL.replace(/\/$/, "") + "/widget_contact?api_key=" + encodeURIComponent(API_KEY) + demoSuffix, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, email, phone, message })
      });
      const text = await res.text();
      if (!res.ok) {
        contactStatus.textContent = text || "Unable to send contact request.";
        contactStatus.style.color = "#b91c1c";
        if (res.status === 403) {
          contactLink.style.display = "none";
          toggleContactForm(false);
        }
        return;
      }
      contactStatus.textContent = "Thanks, we have received your message.";
      contactStatus.style.color = "#15803d";
      nameInput.value = "";
      emailInput.value = "";
      phoneInput.value = "";
      messageInput.value = "";
      setTimeout(() => toggleContactForm(false), 1200);
    } catch (e) {
      contactStatus.textContent = "Network error. Please try again.";
      contactStatus.style.color = "#b91c1c";
    } finally {
      contactSend.disabled = false;
    }
  });

  document.body.appendChild(btn);
  document.body.appendChild(panel);
})(); 
