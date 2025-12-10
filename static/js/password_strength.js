(function() {
  const bar = document.getElementById("pw-strength-bar");
  if (bar) {
    const styleTag = document.createElement("style");
    styleTag.textContent = "body.dark #pw-strength-bar { background: #111827; }";
    document.head.appendChild(styleTag);
  }

  const pwInput = document.getElementById("password");
  const pwConfirm = document.getElementById("confirm_password");
  const matchText = document.getElementById("pw-match-text");
  const strengthFill = document.getElementById("pw-strength-fill");
  const strengthLabel = document.getElementById("pw-strength-label");

  const ruleLength = document.getElementById("pw-rule-length");
  const ruleUpper = document.getElementById("pw-rule-upper");
  const ruleLower = document.getElementById("pw-rule-lower");
  const ruleDigit = document.getElementById("pw-rule-digit");
  const ruleSpecial = document.getElementById("pw-rule-special");

  function setRuleState(el, ok) {
    if (!el) return;
    const icon = el.querySelector(".pw-rule-icon");
    el.style.color = ok ? "#22c55e" : "#6b7280";
    if (icon) {
      icon.textContent = ok ? "✓" : "✘";
    }
  }

  function evaluatePassword(pw) {
    const hasLength = pw.length >= 8;
    const hasUpper = /[A-Z]/.test(pw);
    const hasLower = /[a-z]/.test(pw);
    const hasDigit = /[0-9]/.test(pw);
    const hasSpecial = /[^A-Za-z0-9]/.test(pw);

    let score = 0;
    if (hasLength) score++;
    if (hasUpper) score++;
    if (hasLower) score++;
    if (hasDigit) score++;
    if (hasSpecial) score++;

    let level, width, color;
    if (!pw) {
      level = "Too weak";
      width = "5%";
      color = "#ef4444";
    } else if (score <= 2) {
      level = "Weak";
      width = "25%";
      color = "#ef4444";
    } else if (score === 3 || score === 4) {
      level = "Medium";
      width = "60%";
      color = "#f59e0b";
    } else {
      level = "Strong";
      width = "100%";
      color = "#22c55e";
    }

    return {
      hasLength,
      hasUpper,
      hasLower,
      hasDigit,
      hasSpecial,
      level,
      width,
      color,
    };
  }

  function updateStrength() {
    if (!pwInput) return;
    const pw = pwInput.value || "";
    const result = evaluatePassword(pw);

    if (strengthFill) {
      strengthFill.style.width = result.width;
      strengthFill.style.backgroundColor = result.color;
    }
    if (strengthLabel) {
      strengthLabel.textContent = result.level;
    }

    setRuleState(ruleLength, result.hasLength);
    setRuleState(ruleUpper, result.hasUpper);
    setRuleState(ruleLower, result.hasLower);
    setRuleState(ruleDigit, result.hasDigit);
    setRuleState(ruleSpecial, result.hasSpecial);
  }

  function checkMatch() {
    if (!pwInput || !pwConfirm || !matchText) return;
    if (!pwInput.value && !pwConfirm.value) {
      matchText.textContent = "";
      return;
    }
    if (pwInput.value === pwConfirm.value) {
      matchText.textContent = "Passwords match ✓";
      matchText.style.color = "#22c55e";
    } else {
      matchText.textContent = "Passwords do not match";
      matchText.style.color = "#ef4444";
    }
  }

  if (pwInput) {
    pwInput.addEventListener("input", function () {
      updateStrength();
      checkMatch();
    });
  }
  if (pwConfirm) {
    pwConfirm.addEventListener("input", checkMatch);
  }
  updateStrength();
  checkMatch();
})();
