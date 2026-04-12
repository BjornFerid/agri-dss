import os
import json
import base64
import threading
import random
import time
import io
import joblib
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
import paho.mqtt.client as mqtt
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# ── App setup ──────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

BROKER      = "broker.hivemq.com"
PORT        = 1883
SUB_TOPIC   = "agri/sensors"
PUB_TOPIC   = "agri/decision"

# Shared state
latest_decision = {}
hardware_active = False
hardware_lock   = threading.Lock()

# ── ML Models ──────────────────────────────────────────────────────────────
CROP_CLASSES = [
    "rice", "maize", "chickpea", "kidneybeans", "pigeonpeas",
    "mothbeans", "mungbean", "blackgram", "lentil", "pomegranate",
    "banana", "mango", "grapes", "watermelon", "muskmelon",
    "apple", "orange", "papaya", "coconut", "cotton",
    "jute", "coffee"
]

DISEASE_CLASSES = [
    "Healthy", "Bacterial Blight", "Leaf Rust", "Powdery Mildew",
    "Leaf Spot", "Mosaic Virus", "Early Blight", "Late Blight"
]

# Image transform for ResNet-18
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_models():
    """Load or create placeholder models."""
    # Random Forest – load from disk if present, else create a dummy
    import pickle

    rf_path = "crop_model.pkl"        # ← your actual filename here
    if os.path.exists(rf_path):
        with open(rf_path, "rb") as f:
            rf_model = pickle.load(f)
    else:
        from sklearn.ensemble import RandomForestClassifier
        rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        # Fit with dummy data so predict() works
        X_dummy = np.random.rand(100, 7)
        y_dummy = np.random.randint(0, 22, 100)
        rf_model.fit(X_dummy, y_dummy)
        joblib.dump(rf_model, rf_path)
        print("[ML] Dummy Random Forest created (replace with trained model)")

    # ResNet-18 – load weights if present, else use random-init
    cnn_path = "disease_resnet18.pth"
    cnn_model = models.resnet18(weights=None)
    cnn_model.fc = torch.nn.Linear(cnn_model.fc.in_features, 8)
    if os.path.exists(cnn_path):
        cnn_model.load_state_dict(torch.load(cnn_path, map_location="cpu"))
        print("[ML] ResNet-18 weights loaded")
    else:
        print("[ML] ResNet-18 using random weights (replace with trained .pth)")
    cnn_model.eval()

    return rf_model, cnn_model

rf_model, cnn_model = load_models()

# ── Inference helpers ──────────────────────────────────────────────────────
def predict_crop(payload: dict) -> dict:
    features = np.array([[
        payload["N"], payload["P"], payload["K"],
        payload["pH"], payload["moist"],
        payload["temp"], payload["hum"]
    ]])
    idx   = int(rf_model.predict(features)[0])
    proba = float(rf_model.predict_proba(features)[0][idx])
    return {"crop": CROP_CLASSES[idx], "confidence": round(proba * 100, 2)}

def predict_disease(b64_str: str) -> dict:
    try:
        img_bytes = base64.b64decode(b64_str)
        img       = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tensor    = img_transform(img).unsqueeze(0)
        with torch.no_grad():
            logits = cnn_model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]
        idx   = int(probs.argmax())
        conf  = float(probs[idx])
        return {"disease": DISEASE_CLASSES[idx], "confidence": round(conf * 100, 2)}
    except Exception as e:
        return {"disease": "Unknown", "confidence": 0.0, "error": str(e)}

def run_inference(payload: dict) -> dict:
    crop_result    = predict_crop(payload)
    disease_result = predict_disease(payload.get("image_base64", ""))
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sensor_data": {
            "N": payload["N"], "P": payload["P"], "K": payload["K"],
            "pH": payload["pH"], "moisture": payload["moist"],
            "temperature": payload["temp"], "humidity": payload["hum"]
        },
        "crop_recommendation": crop_result,
        "disease_analysis": disease_result
    }

# ── MQTT ───────────────────────────────────────────────────────────────────
def on_connect(client, userdata, flags, rc):
    print(f"[MQTT] Connected (rc={rc})")
    client.subscribe(SUB_TOPIC)

def on_message(client, userdata, msg):
    global latest_decision, hardware_active
    try:
        payload = json.loads(msg.payload.decode())
        with hardware_lock:
            hardware_active = True
        decision = run_inference(payload)
        latest_decision = decision
        client.publish(PUB_TOPIC, json.dumps(decision))
        print(f"[MQTT] Decision published: {decision['crop_recommendation']}")
    except Exception as e:
        print(f"[MQTT] Error processing message: {e}")

mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

def start_mqtt():
    try:
        mqtt_client.connect(BROKER, PORT, 60)
        mqtt_client.loop_start()
    except Exception as e:
        print(f"[MQTT] Could not connect: {e}")

# ── Simulation thread ──────────────────────────────────────────────────────
def simulate_sensor():
    """Generate random sensor data every 5 seconds when no hardware detected."""
    global latest_decision, hardware_active
    while True:
        time.sleep(5)
        with hardware_lock:
            active = hardware_active
            hardware_active = False  # reset; real message sets it back

        if not active:
            payload = {
                "N":           random.randint(0, 140),
                "P":           random.randint(5, 145),
                "K":           random.randint(5, 205),
                "pH":          round(random.uniform(3.5, 9.5), 1),
                "moist":       random.randint(10, 100),
                "temp":        round(random.uniform(8.0, 44.0), 1),
                "hum":         random.randint(14, 100),
                "image_base64": ""
            }
            decision = run_inference(payload)
            decision["source"] = "simulation"
            latest_decision = decision
            mqtt_client.publish(PUB_TOPIC, json.dumps(decision))
            print(f"[SIM] Simulated → {decision['crop_recommendation']['crop']}")

# ── Flask routes ───────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/latest")
def latest():
    return jsonify(latest_decision)

@app.route("/infer", methods=["POST"])
def infer():
    """Manual inference endpoint — accepts JSON payload directly."""
    payload = request.get_json(force=True)
    decision = run_inference(payload)
    return jsonify(decision)

# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    start_mqtt()
    sim_thread = threading.Thread(target=simulate_sensor, daemon=True)
    sim_thread.start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
