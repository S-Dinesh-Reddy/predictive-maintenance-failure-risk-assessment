document.addEventListener("DOMContentLoaded", () => {
  const data = window.PREDICTION_DATA || {};

  const idle = document.getElementById("riskIdle");
  const result = document.getElementById("riskResult");
  const riskPanel = document.getElementById("riskPanel");

  const gaugeArc = document.getElementById("gaugeArc");
  const gaugeNumber = document.getElementById("gaugeNumber");
  const outProbability = document.getElementById("outProbability");
  const gaugeLevel = document.getElementById("gaugeLevel");
  const machineState = document.getElementById("machineState");

  const form = document.getElementById("paramsForm");
  const analyzeBtn = document.getElementById("analyzeBtn");

  const temperatureMonitor = document.getElementById("mon-temp");
  const torqueMonitor = document.getElementById("mon-torque");
  const speedMonitor = document.getElementById("mon-speed");
  const wearMonitor = document.getElementById("mon-wear");

  const circumference = 2 * Math.PI * 110;

  function setMonitor(element, value, min, max) {
    if (!element) return;

    const state = element.querySelector(".monitor__state");
    const fill = element.querySelector(".bar__fill");

    if (!state || !fill) return;

    let percentage = ((value - min) / (max - min)) * 100;
    percentage = Math.max(0, Math.min(100, percentage));

    fill.style.width = percentage + "%";

    let label = "Normal";
    let level = "normal";

    if (percentage >= 80) {
      label = "Critical";
      level = "critical";
    } else if (percentage >= 55) {
      label = "Moderate";
      level = "moderate";
    }

    state.textContent = label;
    element.dataset.state = level;
  }

  function updateMonitors() {
    const air = parseFloat(
      document.querySelector('[name="air_temperature"]')?.value
    );

    const process = parseFloat(
      document.querySelector('[name="process_temperature"]')?.value
    );

    const speed = parseFloat(
      document.querySelector('[name="rotational_speed"]')?.value
    );

    const torque = parseFloat(
      document.querySelector('[name="torque"]')?.value
    );

    const wear = parseFloat(
      document.querySelector('[name="tool_wear"]')?.value
    );

    if (!Number.isNaN(air) && !Number.isNaN(process)) {
      const temperature = (air + process) / 2;
      setMonitor(temperatureMonitor, temperature, 295, 315);
    }

    if (!Number.isNaN(torque)) {
      setMonitor(torqueMonitor, torque, 10, 80);
    }

    if (!Number.isNaN(speed)) {
      setMonitor(speedMonitor, speed, 1000, 2200);
    }

    if (!Number.isNaN(wear)) {
      setMonitor(wearMonitor, wear, 0, 250);
    }
  }

  function showResult() {
    if (
      data.probability === null ||
      data.probability === undefined ||
      data.probability === ""
    ) {
      if (idle) idle.hidden = false;
      if (result) result.hidden = true;
      return;
    }

    const probability = Math.max(
      0,
      Math.min(100, Number(data.probability))
    );

    if (idle) idle.hidden = true;
    if (result) result.hidden = false;

    // Display formatted percentage (support decimal accuracy for low probabilities like 0.02%)
    const formatted = probability < 1 ? probability.toFixed(2) + "%" : probability.toFixed(1) + "%";

    if (gaugeNumber) {
      gaugeNumber.textContent = formatted;
    }

    if (outProbability) {
      outProbability.textContent = formatted;
    }

    if (gaugeLevel) {
      gaugeLevel.textContent = data.risk_level || "UNKNOWN";
    }

    if (machineState) {
      machineState.textContent =
        Number(data.prediction) === 1
          ? "MAINTENANCE REQUIRED"
          : "NORMAL";
    }

    if (gaugeArc) {
      const offset = circumference * (1 - probability / 100);
      gaugeArc.style.strokeDasharray = `${circumference} ${circumference}`;
      gaugeArc.style.strokeDashoffset = offset;
      gaugeArc.style.transition = "stroke-dashoffset 1.2s cubic-bezier(0.22, 1, 0.36, 1)";
    }

    const risk = String(data.risk_level || "").toLowerCase();

    // Apply risk classes to panel to trigger CSS variable cascading for glows and gauges
    if (riskPanel) {
      riskPanel.classList.remove("is-low", "is-medium", "is-high");
      if (risk) {
        riskPanel.classList.add("is-" + risk);
      }
    }

    if (result) {
      result.dataset.risk = risk;
    }

    if (gaugeLevel) {
      gaugeLevel.dataset.risk = risk;
    }

    if (machineState) {
      machineState.dataset.risk = risk;
    }

    // Activate all flow steps to indicate completed evaluation
    const flowSteps = document.querySelectorAll(".flow__step");
    flowSteps.forEach((step) => {
      step.classList.add("is-active");
    });
  }

  // Live monitor updates as user enters values
  const inputs = document.querySelectorAll(
    '#paramsForm input[type="number"], #paramsForm select'
  );
  inputs.forEach((input) => {
    input.addEventListener("input", updateMonitors);
    input.addEventListener("change", updateMonitors);
  });

  if (form) {
    form.addEventListener("submit", () => {
      if (analyzeBtn) {
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = "ANALYZING...";
      }
    });
  }

  // Generate subtle background ambient particles
  const particlesContainer = document.getElementById("particles");
  if (particlesContainer && particlesContainer.children.length === 0) {
    for (let i = 0; i < 20; i++) {
      const p = document.createElement("div");
      p.className = "particle";
      p.style.left = Math.random() * 100 + "%";
      p.style.top = Math.random() * 100 + "%";
      p.style.animationDuration = 12 + Math.random() * 18 + "s";
      p.style.animationDelay = Math.random() * 8 + "s";
      particlesContainer.appendChild(p);
    }
  }

  updateMonitors();
  showResult();
});