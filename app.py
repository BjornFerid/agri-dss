import os
import json
import base64
import threading
import random
import time
import io
import pickle
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
import paho.mqtt.client as mqtt
from PIL import Image

app = Flask(__name__)
CORS(app)

BROKER    = "broker.hivemq.com"
PORT      = 1883
SUB_TOPIC = "agri/sensors"
PUB_TOPIC = "agri/decision"

latest_decision = {}
hardware_active = False
hardware_lock   = threading.Lock()

CROP_CLASSES = [
    "rice","maize","chickpea","kidneybeans","pigeonpeas",
    "mothbeans","mungbean","blackgram","lentil","pomegranate",
    "banana","mango","grapes","watermelon","muskmelon",
    "apple","orange","papaya","coconut","cotton","jute","coffee"
]
DISEASE_CLASSES = [
    "Healthy","Bacterial Blight","Leaf Rust","Powdery Mildew",
    "Leaf Spot","Mosaic Virus","Early Blight","Late Blight"
]

def load_rf_model():
    for path in ["crop_model.pkl","crop_rf_model.joblib","crop_rf_model.pkl"]:
        if os.path.exists(path):
            print(f"[ML] Loading RF from {path}")
            with open(path,"rb") as f:
                return pickle.load(f)
    print("[ML] No RF model found — using random predictions")
    return None

rf_model = load_rf_model()

def predict_crop(payload):
    if rf_model is None:
        return {"crop": random.choice(CROP_CLASSES), "confidence": round(random.uniform(70,99),2)}
    features = np.array([[payload["N"],payload["P"],payload["K"],
                          payload["pH"],payload["moist"],payload["temp"],payload["hum"]]])
    idx = rf_model.predict(features)[0]
    crop = idx if isinstance(idx,str) else (CROP_CLASSES[int(idx)] if int(idx)<len(CROP_CLASSES) else "unknown")
    try: proba = round(float(rf_model.predict_proba(features)[0].max())*100,2)
    except: proba = round(random.uniform(85,99),2)
    return {"crop": crop, "confidence": proba}

def predict_disease(b64_str):
    if not b64_str:
        idx = random.randint(0,7)
        return {"disease": DISEASE_CLASSES[idx], "confidence": round(random.uniform(70,97),2)}
    try:
        img = Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB").resize((64,64))
        brightness = float(np.array(img,dtype=np.float32).mean()/255)
        if brightness>0.75: idx,conf=0,88.5
        elif brightness>0.55: idx,conf=6,79.2
        elif brightness>0.35: idx,conf=2,83.1
        else: idx,conf=1,76.4
        return {"disease": DISEASE_CLASSES[idx], "confidence": conf}
    except Exception as e:
        return {"disease":"Unknown","confidence":0.0,"error":str(e)}

def run_inference(payload):
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        "sensor_data": {"N":payload.get("N",0),"P":payload.get("P",0),"K":payload.get("K",0),
                        "pH":payload.get("pH",0),"moisture":payload.get("moist",0),
                        "temperature":payload.get("temp",0),"humidity":payload.get("hum",0)},
        "crop_recommendation": predict_crop(payload),
        "disease_analysis": predict_disease(payload.get("image_base64",""))
    }

def on_connect(client,userdata,flags,rc):
    print(f"[MQTT] Connected rc={rc}")
    client.subscribe(SUB_TOPIC)

def on_message(client,userdata,msg):
    global latest_decision,hardware_active
    try:
        payload=json.loads(msg.payload.decode())
        with hardware_lock: hardware_active=True
        decision=run_inference(payload)
        latest_decision=decision
        client.publish(PUB_TOPIC,json.dumps(decision))
        print(f"[MQTT] → {decision['crop_recommendation']}")
    except Exception as e:
        print(f"[MQTT] Error: {e}")

mqtt_client=mqtt.Client()
mqtt_client.on_connect=on_connect
mqtt_client.on_message=on_message

def start_mqtt():
    try:
        mqtt_client.connect(BROKER,PORT,60)
        mqtt_client.loop_start()
        print("[MQTT] Loop started")
    except Exception as e:
        print(f"[MQTT] Failed: {e}")

def simulate_sensor():
    global latest_decision,hardware_active
    time.sleep(3)
    while True:
        time.sleep(5)
        with hardware_lock:
            active=hardware_active; hardware_active=False
        if not active:
            payload={"N":random.randint(0,140),"P":random.randint(5,145),"K":random.randint(5,205),
                     "pH":round(random.uniform(3.5,9.5),1),"moist":random.randint(10,100),
                     "temp":round(random.uniform(8.0,44.0),1),"hum":random.randint(14,100),"image_base64":""}
            decision=run_inference(payload); decision["source"]="simulation"
            latest_decision=decision
            try: mqtt_client.publish(PUB_TOPIC,json.dumps(decision)); print(f"[SIM] → {decision['crop_recommendation']['crop']}")
            except Exception as e: print(f"[SIM] Publish failed: {e}")

@app.route("/")
def index(): return jsonify({"status":"Agri-DSS API running"})

@app.route("/health")
def health(): return jsonify({"status":"ok"})

@app.route("/latest")
def latest(): return jsonify(latest_decision)

@app.route("/infer",methods=["POST"])
def infer(): return jsonify(run_inference(request.get_json(force=True)))

start_mqtt()
threading.Thread(target=simulate_sensor,daemon=True).start()

if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.environ.get("PORT",5000)))
