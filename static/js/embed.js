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
  const ENDPOINT = BASE_URL.replace(/\/$/, "") + "/chat_stream?api_key=" + encodeURIComponent(API_KEY);
  const CONFIG_URL = BASE_URL.replace(/\/$/, "") + "/widget_config?api_key=" + encodeURIComponent(API_KEY);
  const logoUrl = baseUrl
    ? `${baseUrl.replace(/\/+$/, "")}/static/img/theochat-logo-mark.png`
    : "/static/img/theochat-logo-mark.png";

  const DEFAULTS = {
    primary: "#2563eb",
    secondary: "#1d4ed8",
    font: "system-ui, -apple-system, 'Segoe UI', sans-serif",
    radius: "12px",
  };

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
    }
  } catch (e) {
    // ignore config fetch errors; fall back to defaults
  }

  const root = document.documentElement;
  root.style.setProperty("--theochat-primary", theme.primary);
  root.style.setProperty("--theochat-secondary", theme.secondary);
  root.style.setProperty("--theochat-font-family", theme.font);
  root.style.setProperty("--theochat-radius", theme.radius);

  const btn = document.createElement("button");
  const defaultLabel = "Chat with us";
  const customLabel = scriptEl && scriptEl.getAttribute("data-button-label");
  btn.textContent = customLabel || defaultLabel;
  btn.style.position = "fixed";
  btn.style.bottom = isDemo ? "24px" : "16px";
  btn.style.right = isDemo ? "24px" : "16px";
  btn.style.zIndex = 9999;
  btn.style.background = "var(--theochat-primary, #2563eb)";
  btn.style.color = "#fff";
  btn.style.border = "none";
  btn.style.borderRadius = "999px";
  btn.style.padding = "10px 14px";
  btn.style.boxShadow = "0 4px 12px rgba(0,0,0,0.15)";
  btn.style.cursor = "pointer";
  btn.style.fontFamily = "var(--theochat-font-family, " + DEFAULTS.font + ")";

  const panel = document.createElement("div");
  panel.style.position = "fixed";
  panel.style.bottom = isDemo ? "70px" : "60px";
  panel.style.right = isDemo ? "24px" : "16px";
  panel.style.width = "320px";
  panel.style.height = "470px";
  panel.style.background = "#fff";
  panel.style.color = "#111";
  panel.style.border = "1px solid #e5e7eb";
  panel.style.borderRadius = "var(--theochat-radius, 12px)";
  panel.style.boxShadow = "0 10px 30px rgba(0,0,0,0.25)";
  panel.style.display = "none";
  panel.style.flexDirection = "column";
  panel.style.overflow = "hidden";
  panel.style.fontFamily = "var(--theochat-font-family, " + DEFAULTS.font + ")";
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
  }

  function closePanel() {
    panel.style.opacity = "0";
    panel.style.transform = "translateY(8px)";
    setTimeout(() => {
      panel.style.display = "none";
    }, 200);
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
  inputRow.style.gap = "6px";
  inputRow.style.padding = "10px";
  inputRow.style.borderTop = "1px solid #e5e7eb";
  const input = document.createElement("input");
  input.type = "text";
  input.placeholder = "Ask Theo a question about this site...";
  input.style.flex = "1";
  input.style.fontSize = "14px";
  input.style.padding = "8px";
  input.style.border = "1px solid #d1d5db";
  input.style.borderRadius = "var(--theochat-radius, 8px)";
  input.style.fontFamily = "var(--theochat-font-family, " + DEFAULTS.font + ")";
  const sendBtn = document.createElement("button");
  sendBtn.textContent = "Send";
  sendBtn.style.padding = "8px 12px";
  sendBtn.style.border = "none";
  sendBtn.style.borderRadius = "var(--theochat-radius, 8px)";
  sendBtn.style.background = "var(--theochat-primary, #2563eb)";
  sendBtn.style.color = "#fff";
  sendBtn.style.cursor = "pointer";
  sendBtn.style.fontFamily = "var(--theochat-font-family, " + DEFAULTS.font + ")";

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

  const container = document.createElement("div");
  container.style.display = "flex";
  container.style.flexDirection = "column";
  container.appendChild(header);
  container.appendChild(chatArea);
  container.appendChild(contactWrapper);
  container.appendChild(inputRow);

  panel.appendChild(container);
  panel.appendChild(resizer);

  btn.addEventListener("click", () => {
    const isHidden = panel.style.display === "none";
    if (isHidden) {
      openPanel();
    } else {
      closePanel();
    }
  });

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
      const res = await fetch(BASE_URL.replace(/\/$/, "") + "/widget_contact?api_key=" + encodeURIComponent(API_KEY), {
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
