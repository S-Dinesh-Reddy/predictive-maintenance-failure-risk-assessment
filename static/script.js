/**
 * INDUSTRIAL PREDICTIVE MAINTENANCE CONTROL CENTER
 * Client-Side Telemetry, Two-Level Validation & Live Prediction Interface Controller
 * 
 * Level 1: Hard Sanity Limits (Reject physically impossible / nonsensical inputs)
 * Level 2: Training Distribution Warning (Allow extrapolation with soft advisory banner)
 */

document.addEventListener("DOMContentLoaded", () => {
  const data = window.PREDICTION_DATA || {};

  // Form & Action Controls
  const form = document.getElementById("paramsForm");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const analyzeBtnLabel = document.getElementById("analyzeBtnLabel");

  const inputAir = document.getElementById("inputAirTemp");
  const inputProc = document.getElementById("inputProcTemp");
  const inputSpeed = document.getElementById("inputSpeed");
  const inputTorque = document.getElementById("inputTorque");
  const inputWear = document.getElementById("inputWear");
  const selectType = document.getElementById("selectType");

  // Hero & Live Status Elements
  const overviewTypeBadge = document.getElementById("overviewTypeBadge");
  const heroAnalysisState = document.getElementById("heroAnalysisState");
  const liveTimestamp = document.getElementById("liveTimestamp");

  // Preset Controls
  const presetNormalBtn = document.getElementById("presetNormalBtn");
  const presetRiskBtn = document.getElementById("presetRiskBtn");

  // Prediction Output Display Elements
  const outProbability = document.getElementById("outProbability");
  const probBarFill = document.getElementById("probBarFill");
  const outRiskLevel = document.getElementById("outRiskLevel");
  const outMachineStatus = document.getElementById("outMachineStatus");
  const outInterpretationText = document.getElementById("outInterpretationText");
  const distributionWarningBanner = document.getElementById("distributionWarningBanner");
  const distributionWarningText = document.getElementById("distributionWarningText");

  // Recommended Action Elements
  const actionHeaderTitle = document.getElementById("actionHeaderTitle");
  const actionAlertBadge = document.querySelector(".action-alert-badge");
  const actionPrimaryText = document.getElementById("actionPrimaryText");
  const actionStep1 = document.getElementById("actionStep1");
  const actionStep2 = document.getElementById("actionStep2");
  const actionStep3 = document.getElementById("actionStep3");

  // LEVEL 1: Hard Application Sanity Limits (Prevents nonsensical/impossible inputs)
  const SANITY_CONFIG = {
    inputAirTemp: {
      name: "air_temperature",
      label: "Air Temperature",
      min: 200,
      max: 400,
      unit: "K",
      allowDecimal: true,
      msgId: "msgAirTemp",
      extId: "extAirTemp"
    },
    inputProcTemp: {
      name: "process_temperature",
      label: "Process Temperature",
      min: 200,
      max: 450,
      unit: "K",
      allowDecimal: true,
      msgId: "msgProcTemp",
      extId: "extProcTemp"
    },
    inputSpeed: {
      name: "rotational_speed",
      label: "Rotational Speed",
      min: 100,
      max: 6000,
      unit: "rpm",
      allowDecimal: false,
      msgId: "msgSpeed",
      extId: "extSpeed"
    },
    inputTorque: {
      name: "torque",
      label: "Torque",
      min: 0,
      max: 150,
      unit: "Nm",
      allowDecimal: true,
      msgId: "msgTorque",
      extId: "extTorque"
    },
    inputWear: {
      name: "tool_wear",
      label: "Tool Wear",
      min: 0,
      max: 500,
      unit: "min",
      allowDecimal: false,
      msgId: "msgWear",
      extId: "extWear"
    }
  };

  // LEVEL 2: Observed AI4I 2020 Model Training Distribution (For advisory extrapolation notice)
  const TRAINING_DISTRIBUTION = {
    air_temperature: { min: 295.3, max: 304.5, unit: "K" },
    process_temperature: { min: 305.7, max: 313.8, unit: "K" },
    rotational_speed: { min: 1168, max: 2886, unit: "rpm" },
    torque: { min: 3.8, max: 76.6, unit: "Nm" },
    tool_wear: { min: 0, max: 253, unit: "min" }
  };

  /**
   * Live Clock for Header Timestamp
   */
  function updateLiveClock() {
    if (!liveTimestamp) return;
    const now = new Date();
    const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    const day = now.getDate();
    const month = months[now.getMonth()];
    const year = now.getFullYear();
    const hours = String(now.getHours()).padStart(2, "0");
    const minutes = String(now.getMinutes()).padStart(2, "0");
    const seconds = String(now.getSeconds()).padStart(2, "0");
    liveTimestamp.textContent = `${day} ${month} ${year} ${hours}:${minutes}:${seconds}`;
  }
  updateLiveClock();
  setInterval(updateLiveClock, 1000);

  /**
   * Update Machine Variant Label on Select Change
   */
  function updateTypeLabel() {
    if (!overviewTypeBadge || !selectType) return;
    const val = selectType.value;
    if (val === "H") overviewTypeBadge.textContent = "H — High Quality Variant";
    else if (val === "M") overviewTypeBadge.textContent = "M — Medium Quality Variant";
    else if (val === "L") overviewTypeBadge.textContent = "L — Low Quality Variant";
  }
  if (selectType) {
    selectType.addEventListener("change", updateTypeLabel);
  }

  /**
   * Strict Keydown Filter: Prohibits non-numeric & malformed keystrokes at typing time
   */
  function handleKeyDown(e, config) {
    const allowedKeys = [
      "Backspace", "Delete", "Tab", "Escape", "Enter",
      "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown",
      "Home", "End"
    ];
    if (allowedKeys.includes(e.key)) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;

    // Allow digits 0-9
    if (/^[0-9]$/.test(e.key)) return;

    // Allow single decimal point if permitted and not already present
    if (config.allowDecimal && e.key === ".") {
      if (!e.target.value.includes(".")) {
        return;
      }
    }

    // Block everything else: minus (-), plus (+), 'e', 'E', letters, punctuation
    e.preventDefault();
  }

  /**
   * Two-Level Field Validator:
   * Level 1: Hard Sanity check (rejects if out of sanity bounds)
   * Level 2: Training Range check (advisory soft warning if extrapolating)
   */
  function validateField(inputEl, config) {
    const card = inputEl.closest(".sensor-card");
    const msgEl = config.msgId ? document.getElementById(config.msgId) : null;
    const extEl = config.extId ? document.getElementById(config.extId) : null;
    const tDist = TRAINING_DISTRIBUTION[config.name];

    // 1. Browser badInput flag
    if (inputEl.validity && inputEl.validity.badInput) {
      setFieldInvalid(card, msgEl, extEl, `⚠ Invalid ${config.label} format.`);
      return false;
    }

    const rawVal = inputEl.value;
    if (rawVal === null || rawVal === undefined || rawVal.trim() === "") {
      setFieldInvalid(card, msgEl, extEl, `⚠ ${config.label} is required.`);
      return false;
    }

    const trimmed = rawVal.trim();

    // 2. Strict numeric regex
    const strictPattern = config.allowDecimal ? /^\d+(\.\d+)?$/ : /^\d+$/;
    if (!strictPattern.test(trimmed)) {
      if (trimmed.includes("-")) {
        setFieldInvalid(card, msgEl, extEl, `⚠ Negative values unsupported (Supported: ${config.min}–${config.max} ${config.unit}).`);
      } else {
        setFieldInvalid(card, msgEl, extEl, `⚠ Enter valid number (Supported: ${config.min}–${config.max} ${config.unit}).`);
      }
      return false;
    }

    const num = Number(trimmed);

    // 3. Finite number check
    if (!Number.isFinite(num)) {
      setFieldInvalid(card, msgEl, extEl, `⚠ Enter a valid finite number.`);
      return false;
    }

    // 4. LEVEL 1: Hard Sanity Range Check (blocks invalid inputs)
    if (num < config.min || num > config.max) {
      setFieldInvalid(card, msgEl, extEl, `⚠ Supported: ${config.min} – ${config.max} ${config.unit}.`);
      return false;
    }

    // Input is Level 1 Valid!
    setFieldValid(card, msgEl);

    // 5. LEVEL 2: Training Distribution Check (advisory notice, does NOT block)
    if (tDist && (num < tDist.min || num > tDist.max)) {
      setFieldExtrapolating(card, extEl, `⚡ Outside training range (${tDist.min}–${tDist.max} ${config.unit})`);
    } else {
      clearFieldExtrapolating(card, extEl);
    }

    return true;
  }

  function setFieldInvalid(card, msgEl, extEl, text) {
    if (card) {
      card.classList.add("is-invalid");
      card.classList.remove("is-extrapolating");
    }
    if (msgEl) msgEl.textContent = text;
    if (extEl) extEl.textContent = "";
  }

  function setFieldValid(card, msgEl) {
    if (card) card.classList.remove("is-invalid");
    if (msgEl) msgEl.textContent = "";
  }

  function setFieldExtrapolating(card, extEl, text) {
    if (card) card.classList.add("is-extrapolating");
    if (extEl) extEl.textContent = text;
  }

  function clearFieldExtrapolating(card, extEl) {
    if (card) card.classList.remove("is-extrapolating");
    if (extEl) extEl.textContent = "";
  }

  /**
   * Validate All Form Fields against Level 1 Hard Sanity Bounds
   */
  function validateAllInputs() {
    let allValid = true;
    Object.keys(SANITY_CONFIG).forEach((id) => {
      const el = document.getElementById(id);
      const config = SANITY_CONFIG[id];
      if (el) {
        const isValid = validateField(el, config);
        if (!isValid) allValid = false;
      }
    });

    if (selectType && !["H", "M", "L"].includes(selectType.value)) {
      allValid = false;
    }

    return allValid;
  }

  /**
   * Invalidate / Clear Previous Prediction Results ONLY When Inputs Become Hard Invalid
   */
  function invalidatePredictionResults(reason = "Awaiting valid machine parameters.") {
    if (outProbability) outProbability.textContent = "--";
    if (probBarFill) {
      probBarFill.style.width = "0%";
      probBarFill.style.background = "#6b7280";
    }
    if (outRiskLevel) {
      outRiskLevel.textContent = "AWAITING INPUT";
      outRiskLevel.className = "risk-pill-badge badge-neutral";
    }
    if (outMachineStatus) {
      outMachineStatus.textContent = "AWAITING INPUT";
      outMachineStatus.className = "machine-status-pill pill-neutral";
    }
    if (outInterpretationText) {
      outInterpretationText.textContent = reason;
    }
    if (heroAnalysisState) {
      heroAnalysisState.textContent = "AWAITING VALID PARAMETERS";
    }
    if (actionHeaderTitle) {
      actionHeaderTitle.textContent = "AWAITING VALID TELEMETRY";
    }
    if (actionAlertBadge) {
      actionAlertBadge.className = "action-alert-badge badge-neutral";
    }
    if (actionPrimaryText) {
      actionPrimaryText.textContent = reason;
    }
    if (actionStep1) {
      actionStep1.textContent = "Enter operating telemetry within supported machine bounds.";
    }
    if (actionStep2) {
      actionStep2.textContent = "Verify all 5 sensor parameters show no hard validation errors.";
    }
    if (actionStep3) {
      actionStep3.textContent = "Click 'ANALYZE MACHINE' to execute XGBoost risk classification.";
    }
    if (distributionWarningBanner) {
      distributionWarningBanner.style.display = "none";
    }
    const summarySection = document.getElementById("telemetry-summary");
    if (summarySection) {
      summarySection.style.opacity = "0.45";
    }
  }

  /**
   * Render Valid Model Prediction Results & Training Warnings
   */
  function renderPredictionResults() {
    if (data.probability === null || data.probability === undefined || data.probability === "") {
      if (heroAnalysisState) heroAnalysisState.textContent = "READY FOR INPUT";
      return;
    }

    const probVal = Math.max(0, Math.min(100, Number(data.probability)));
    const riskLevel = String(data.risk_level || "LOW").toUpperCase();
    const predictionClass = Number(data.prediction);

    const formattedProb = probVal < 1 ? probVal.toFixed(2) + "%" : probVal.toFixed(1) + "%";

    if (outProbability) outProbability.textContent = formattedProb;

    // Progress Bar Fill & Semantic Color
    if (probBarFill) {
      probBarFill.style.width = Math.max(1.5, probVal) + "%";
      if (riskLevel === "HIGH") {
        probBarFill.style.background = "#ef4444";
      } else if (riskLevel === "MEDIUM") {
        probBarFill.style.background = "#f59e0b";
      } else {
        probBarFill.style.background = "#10b981";
      }
    }

    // Risk Level Badge Styling
    if (outRiskLevel) {
      outRiskLevel.textContent = riskLevel;
      outRiskLevel.className = "risk-pill-badge badge-" + riskLevel.toLowerCase();
    }

    // Machine Status Pill Styling
    if (outMachineStatus) {
      const isFailed = predictionClass === 1;
      outMachineStatus.textContent = isFailed ? "MAINTENANCE REQUIRED" : "NORMAL";
      outMachineStatus.className = "machine-status-pill pill-" + (isFailed ? "high" : "low");
    }

    // Training Warning Banner
    if (distributionWarningBanner) {
      if (data.training_warning) {
        distributionWarningBanner.style.display = "flex";
        if (distributionWarningText && data.out_of_dist_features && data.out_of_dist_features.length > 0) {
          distributionWarningText.textContent = `Operating conditions (${data.out_of_dist_features.join(', ')}) are outside the observed AI4I 2020 training distribution. Prediction is provided, but model confidence may be reduced due to extrapolation.`;
        }
      } else {
        distributionWarningBanner.style.display = "none";
      }
    }

    // Update Hero state
    if (heroAnalysisState) {
      heroAnalysisState.textContent = `EVALUATED · ${riskLevel} RISK`;
    }

    const summarySection = document.getElementById("telemetry-summary");
    if (summarySection) {
      summarySection.style.opacity = "1";
    }
  }

  /**
   * Attach Real-time Input Listeners
   */
  Object.keys(SANITY_CONFIG).forEach((id) => {
    const el = document.getElementById(id);
    const config = SANITY_CONFIG[id];
    if (el) {
      el.addEventListener("keydown", (e) => handleKeyDown(e, config));

      el.addEventListener("input", () => {
        const valid = validateField(el, config);
        if (!valid) {
          // Hard invalid -> clear old results
          invalidatePredictionResults("Awaiting valid machine parameters.");
        }
      });

      el.addEventListener("blur", () => {
        validateField(el, config);
      });
    }
  });

  /**
   * Load Test Bench Presets
   */
  function loadPreset(air, proc, speed, torque, wear, type) {
    if (inputAir) inputAir.value = air;
    if (inputProc) inputProc.value = proc;
    if (inputSpeed) inputSpeed.value = speed;
    if (inputTorque) inputTorque.value = torque;
    if (inputWear) inputWear.value = wear;
    if (selectType) selectType.value = type;

    updateTypeLabel();

    // Clear all invalid markers and check training ranges
    Object.keys(SANITY_CONFIG).forEach((id) => {
      const el = document.getElementById(id);
      const config = SANITY_CONFIG[id];
      if (el) validateField(el, config);
    });

    const conditionsSection = document.getElementById("operating-conditions");
    if (conditionsSection) {
      conditionsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  if (presetNormalBtn) {
    presetNormalBtn.addEventListener("click", () => {
      loadPreset(300, 310, 1400, 30, 20, "H");
    });
  }

  if (presetRiskBtn) {
    presetRiskBtn.addEventListener("click", () => {
      loadPreset(300, 310, 1300, 65, 200, "L");
    });
  }

  /**
   * Form Submission Handling & Validation Guard
   */
  if (form) {
    form.addEventListener("submit", (e) => {
      const isFormValid = validateAllInputs();
      if (!isFormValid) {
        e.preventDefault();
        invalidatePredictionResults("Cannot analyze: One or more inputs are outside the supported operating bounds.");
        const firstInvalid = document.querySelector(".sensor-card.is-invalid input");
        if (firstInvalid) firstInvalid.focus();
        return false;
      }

      if (analyzeBtn) {
        analyzeBtn.disabled = true;
        if (analyzeBtnLabel) {
          analyzeBtnLabel.textContent = "ANALYZING...";
        }
      }
    });
  }

  // Initialize on Load
  updateTypeLabel();
  renderPredictionResults();
});