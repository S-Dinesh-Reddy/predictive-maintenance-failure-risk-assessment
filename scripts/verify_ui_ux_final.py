"""
UI/UX and Functional Verification Script
Validates the Flask application, UI templates, CSS tokens, JavaScript controllers,
and API prediction consistency under various simulated telemetry scenarios.
"""
import urllib.request
import urllib.parse
import json
import sys

def run_tests():
    base_url = "http://127.0.0.1:5000"
    print(f"Connecting to live Flask server at {base_url}...")

    # 1. GET Homepage / Dashboard
    with urllib.request.urlopen(f"{base_url}/") as res:
        assert res.status == 200, f"Expected HTTP 200, got {res.status}"
        html = res.read().decode("utf-8")
        assert "PREDICTIVE MAINTENANCE" in html, "Missing header title"
        assert "AWAITING TELEMETRY" in html, "Initial unsubmitted state incorrect"
        assert 'id="analyzeBtn"' in html, "Missing analyze button"
        assert "aria-describedby" in html, "Missing accessibility describedby bindings"
        assert "aria-live=\"polite\"" in html, "Missing live region for screen readers"
        assert "BIVARIATE DEPENDENCE" in html, "Missing SHAP section categorization"
        assert "Statistical attribution, not physical causation" in html, "Missing SHAP governance disclaimer"
        print("[PASS] 1. Initial Dashboard HTML renders cleanly with proper accessibility and initial state.")

    # 2. POST /predict with Nominal Operating Conditions
    params_normal = {
        "air_temperature": "300.0",
        "process_temperature": "310.0",
        "rotational_speed": "1400",
        "torque": "30.0",
        "tool_wear": "20",
        "machine_type": "H"
    }
    data_normal = urllib.parse.urlencode(params_normal).encode("utf-8")
    req_norm = urllib.request.Request(f"{base_url}/predict", data=data_normal, method="POST")
    with urllib.request.urlopen(req_norm) as res_norm:
        assert res_norm.status == 200
        norm_html = res_norm.read().decode("utf-8")
        assert "LOW RISK" in norm_html, "Expected LOW RISK for nominal inputs"
        assert "NORMAL" in norm_html, "Expected NORMAL machine status for nominal inputs"
        print("[PASS] 2. Nominal Telemetry submission yields LOW RISK and NORMAL machine status.")

    # 3. POST /predict with High-Stress Operating Conditions
    params_stress = {
        "air_temperature": "303.0",
        "process_temperature": "312.0",
        "rotational_speed": "1250",
        "torque": "68.0",
        "tool_wear": "230",
        "machine_type": "L"
    }
    data_stress = urllib.parse.urlencode(params_stress).encode("utf-8")
    req_stress = urllib.request.Request(f"{base_url}/predict", data=data_stress, method="POST")
    with urllib.request.urlopen(req_stress) as res_stress:
        assert res_stress.status == 200
        stress_html = res_stress.read().decode("utf-8")
        assert "HIGH RISK" in stress_html, "Expected HIGH RISK for high-stress inputs"
        assert "MAINTENANCE REQUIRED" in stress_html, "Expected MAINTENANCE REQUIRED for high-stress inputs"
        print("[PASS] 3. High-Stress Telemetry submission yields HIGH RISK and MAINTENANCE REQUIRED status.")

    # 4. Verify static/style.css
    with urllib.request.urlopen(f"{base_url}/static/style.css") as res_css:
        assert res_css.status == 200
        css = res_css.read().decode("utf-8")
        assert ":focus-visible" in css, "Missing :focus-visible rules in CSS"
        assert "--text-dim: #8da2b5;" in css, "Missing WCAG AA compliant text-dim color"
        assert "@media (max-width: 520px)" in css, "Missing mobile viewport media query"
        assert ".btn-quick-preset.is-active-preset" in css, "Missing active preset button style"
        assert ".btn-analyze-orange.is-loading" in css, "Missing analyze button loading style"
        print("[PASS] 4. static/style.css verified: WCAG AA contrast, focus rings, loading & active states, mobile breakpoint.")

    # 5. Verify static/script.js
    with urllib.request.urlopen(f"{base_url}/static/script.js") as res_js:
        assert res_js.status == 200
        js = res_js.read().decode("utf-8")
        assert "is-active-preset" in js, "Missing preset active indicator logic"
        assert "clearPresetHighlights" in js, "Missing clear preset highlights function"
        assert "is-loading" in js, "Missing loading button state logic"
        print("[PASS] 5. static/script.js verified: preset indicators and loading submission feedback active.")

    # 6. JSON API validation
    api_payload = json.dumps(params_stress).encode("utf-8")
    req_api = urllib.request.Request(
        f"{base_url}/api/predict",
        data=api_payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req_api) as res_api:
        assert res_api.status == 200
        api_data = json.loads(res_api.read().decode("utf-8"))
        assert api_data["status"] == "success"
        assert api_data["prediction"] == 1
        assert api_data["risk_level"] == "HIGH"
        assert api_data["failure_probability"] >= 50.0
        print(f"[PASS] 6. REST API /api/predict validated (Risk: {api_data['risk_level']}, Prob: {api_data['failure_probability']}%, Class: {api_data['prediction']}).")

    print("\nALL UI/UX AND FUNCTIONAL VERIFICATION CHECKS PASSED PERFECTLY (6/6)!")

if __name__ == "__main__":
    run_tests()
