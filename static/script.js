/* ==========================================================================
   Predictive Maintenance dashboard — presentation layer only.
   No prediction logic lives here; values come from the Flask backend via
   window.PREDICTION_DATA (rendered by Jinja in templates/index.html).
   ========================================================================== */

(function () {
  "use strict";

  /* ----------------------------- background ----------------------------- */

  function spawnParticles() {
    var host = document.getElementById("particles");
    if (!host) return;
    var count = window.innerWidth < 768 ? 12 : 24;
    var frag = document.createDocumentFragment();
    for (var i = 0; i < count; i++) {
      var p = document.createElement("span");
      p.className = "particle";
      p.style.left = Math.random() * 100 + "%";
      p.style.top = 100 + Math.random() * 20 + "%";
      p.style.animationDuration = 16 + Math.random() * 22 + "s";
      p.style.animationDelay = "-" + Math.random() * 25 + "s";
      p.style.opacity = String(0.25 + Math.random() * 0.5);
      frag.appendChild(p);
    }
    host.appendChild(frag);
  }

  /* -------------------------------- gauge ------------------------------- */

  var RADIUS = 110;
  var CIRC = 2 * Math.PI * RADIUS;

  function riskClassFor(level, probability) {
    var l = (level || "").toString().toLowerCase();
    if (l.indexOf("high") > -1) return "is-high";
    if (l.indexOf("medium") > -1 || l.indexOf("moderate") > -1) return "is-medium";
    if (l.indexOf("low") > -1) return "is-low";
    if (probability >= 60) return "is-high";
    if (probability >= 30) return "is-medium";
    return "is-low";
  }

  function setFlowProgress(done) {
    var steps = document.querySelectorAll(".flow__step");
    for (var i = 0; i < steps.length; i++) {
      steps[i].classList.toggle("is-active", done ? true : i === 0);
    }
  }

  function renderRisk(data) {
    var idle = document.getElementById("riskIdle");
    var result = document.getElementById("riskResult");
    var risk = document.getElementById("riskPanel");
    if (!risk || !idle || !result) return;

    if (!data || data.probability === null || data.probability === undefined) {
      idle.hidden = false;
      result.hidden = true;
      setFlowProgress(false);
      return;
    }

    var probability = Math.max(0, Math.min(100, Number(data.probability)));
    var level =
      data.risk_level || (probability >= 60 ? "HIGH" : probability >= 30 ? "MEDIUM" : "LOW");
    var state =
      data.machine_status ||
      (Number(data.prediction) === 1 ? "MAINTENANCE REQUIRED" : "NORMAL");

    idle.hidden = true;
    result.hidden = false;
    setFlowProgress(true);

    risk.classList.remove("is-low", "is-medium", "is-high");
    risk.classList.add(riskClassFor(level, probability));

    var pctText = probability.toFixed(1) + "%";
    document.getElementById("gaugeNumber").textContent = pctText;
    document.getElementById("outProbability").textContent = pctText;
    document.getElementById("gaugeLevel").textContent = level.toString().toUpperCase();
    document.getElementById("machineState").textContent = state.toString().toUpperCase();

    var arc = document.getElementById("gaugeArc");
    arc.style.strokeDasharray = CIRC + " " + CIRC;
    arc.style.strokeDashoffset = String(CIRC);
    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        arc.style.strokeDashoffset = String(CIRC * (1 - probability / 100));
      });
    });
  }

  /* ------------------------------ monitors ------------------------------ */
  /* Purely visual gauges derived from the entered parameters (AI4I typical
     operating envelopes). These are NOT model outputs. */

  var MONITORS = [
    { id: "mon-temp", input: "process_temperature", min: 305, max: 314 },
    { id: "mon-torque", input: "torque", min: 3, max: 77 },
    { id: "mon-speed", input: "rotational_speed", min: 1168, max: 2886 },
    { id: "mon-wear", input: "tool_wear", min: 0, max: 253 },
  ];

  function labelFor(pct) {
    if (pct >= 80) return "Critical";
    if (pct >= 55) return "Moderate";
    return "Normal";
  }

  function stateFor(pct) {
    if (pct >= 80) return "critical";
    if (pct >= 55) return "moderate";
    return "normal";
  }

  function syncMonitors() {
    MONITORS.forEach(function (m) {
      var card = document.getElementById(m.id);
      var input = document.querySelector('[name="' + m.input + '"]');
      if (!card) return;
      var raw = input && input.value !== "" ? Number(input.value) : NaN;
      var pct = isNaN(raw) ? 0 : ((raw - m.min) / (m.max - m.min)) * 100;
      pct = Math.max(0, Math.min(100, pct));
      card.querySelector(".bar__fill").style.width = pct + "%";
      card.querySelector(".monitor__state").textContent = isNaN(raw) ? "No data" : labelFor(pct);
      card.setAttribute("data-state", isNaN(raw) ? "idle" : stateFor(pct));
    });
  }

  /* --------------------------------- init -------------------------------- */

  function init() {
    // Exposed so the result panel can be re-rendered with backend values.
    window.PM_RENDER_RISK = renderRisk;
    spawnParticles();
    renderRisk(window.PREDICTION_DATA || null);
    syncMonitors();

    document.querySelectorAll(".field__control input").forEach(function (el) {
      el.addEventListener("input", syncMonitors);
    });

    var form = document.getElementById("paramsForm");
    if (form) {
      form.addEventListener("submit", function () {
        var btn = document.getElementById("analyzeBtn");
        if (btn) {
          btn.disabled = true;
          btn.textContent = "ANALYZING…";
        }
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
