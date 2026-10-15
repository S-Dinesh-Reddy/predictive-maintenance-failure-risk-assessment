import time
import urllib.request
import json
import subprocess
import sys
import os

print("Starting Flask app in subprocess for live smoke test...")
proc = subprocess.Popen(
    [sys.executable, "app.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

time.sleep(2.5)

try:
    # 1. GET Homepage
    req_home = urllib.request.urlopen("http://127.0.0.1:5000/")
    assert req_home.status == 200
    html = req_home.read().decode('utf-8')
    assert "PREDICTIVE MAINTENANCE" in html
    assert "OPERATING CONDITIONS" in html
    print("[PASS] GET / -> HTTP 200 (Dashboard UI loaded)")

    # 2. GET API Metadata
    req_meta = urllib.request.urlopen("http://127.0.0.1:5000/api/metadata")
    assert req_meta.status == 200
    meta_json = json.loads(req_meta.read().decode('utf-8'))
    assert meta_json["status"] == "success"
    assert meta_json["operational_threshold"] == 0.50
    assert "test_metrics" in meta_json
    print(f"[PASS] GET /api/metadata -> HTTP 200 (Threshold={meta_json['operational_threshold']})")

    # 3. POST API Predict Normal Case
    normal_payload = json.dumps({
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1400.0,
        "torque": 30.0,
        "tool_wear": 20.0,
        "machine_type": "H"
    }).encode('utf-8')
    req_pred = urllib.request.Request(
        "http://127.0.0.1:5000/api/predict",
        data=normal_payload,
        headers={"Content-Type": "application/json"}
    )
    res_pred = urllib.request.urlopen(req_pred)
    assert res_pred.status == 200
    pred_data = json.loads(res_pred.read().decode('utf-8'))
    assert pred_data["status"] == "success"
    assert pred_data["prediction"] == 0
    assert pred_data["risk_level"] == "LOW"
    print(f"[PASS] POST /api/predict (Normal) -> Pred: {pred_data['prediction']}, Risk: {pred_data['risk_level']}, Prob: {pred_data['failure_probability']}%")

    # 4. POST API Predict High-Stress Case
    stress_payload = json.dumps({
        "air_temperature": 303.0,
        "process_temperature": 312.0,
        "rotational_speed": 1250.0,
        "torque": 68.0,
        "tool_wear": 230.0,
        "machine_type": "L"
    }).encode('utf-8')
    req_stress = urllib.request.Request(
        "http://127.0.0.1:5000/api/predict",
        data=stress_payload,
        headers={"Content-Type": "application/json"}
    )
    res_stress = urllib.request.urlopen(req_stress)
    assert res_stress.status == 200
    stress_data = json.loads(res_stress.read().decode('utf-8'))
    assert stress_data["status"] == "success"
    assert stress_data["prediction"] == 1
    assert stress_data["risk_level"] == "HIGH"
    print(f"[PASS] POST /api/predict (Stress) -> Pred: {stress_data['prediction']}, Risk: {stress_data['risk_level']}, Prob: {stress_data['failure_probability']}%")

    print("\nALL LIVE FLASK SMOKE TESTS PASSED CLEANLY!")
finally:
    proc.terminate()
    proc.wait()
    print("Flask server process terminated.")
