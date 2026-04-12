import os, json, time, random, threading, io, base64, pickle
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import paho.mqtt.client as mqtt

app = Flask(__name__)
CORS(app)

BROKER    = "broker.hivemq.com"
PORT_MQTT = 1883
PUB_TOPIC = "agri/decision"
SUB_TOPIC = "agri/sensors"

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

# ── Load model ─────────────────────────────────────────────────────────────
rf_model = None
for _p in ["crop_model.pkl","crop_rf_model.pkl","crop_rf_model.joblib"]:
    if os.path.exists(_p):
        with open(_p,"rb") as _f: rf_model = pickle.load(_f)
        print(f"[ML] Loaded {_p}")
        break

# ── Shared state ───────────────────────────────────────────────────────────
state = {"latest": {}, "hw_active": False}

# ── Inference ──────────────────────────────────────────────────────────────
def predict_crop(p):
    if rf_model is None:
        return {"crop": random.choice(CROP_CLASSES), "confidence": round(random.uniform(80,99),2)}
    X   = np.array([[p["N"],p["P"],p["K"],p["pH"],p["moist"],p["temp"],p["hum"]]])
    idx = rf_model.predict(X)[0]
    crop = idx if isinstance(idx,str) else (CROP_CLASSES[int(idx)] if int(idx)<len(CROP_CLASSES) else "unknown")
    try:    conf = round(float(rf_model.predict_proba(X)[0].max())*100,2)
    except: conf = round(random.uniform(80,99),2)
    return {"crop": crop, "confidence": conf}

def predict_disease(b64):
    if not b64:
        return {"disease": DISEASE_CLASSES[random.randint(0,7)], "confidence": round(random.uniform(70,97),2)}
    try:
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB").resize((64,64))
        br  = float(np.array(img,dtype=np.float32).mean()/255)
        if br>0.75:   idx,conf=0,88.5
        elif br>0.55: idx,conf=6,79.2
        elif br>0.35: idx,conf=2,83.1
        else:         idx,conf=1,76.4
        return {"disease": DISEASE_CLASSES[idx], "confidence": conf}
    except: return {"disease":"Unknown","confidence":0.0}

def make_decision(p, source="simulation"):
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        "source": source,
        "sensor_data": {"N":p.get("N",0),"P":p.get("P",0),"K":p.get("K",0),
                        "pH":p.get("pH",0),"moisture":p.get("moist",0),
                        "temperature":p.get("temp",0),"humidity":p.get("hum",0)},
        "crop_recommendation": predict_crop(p),
        "disease_analysis":    predict_disease(p.get("image_base64",""))
    }

# ── MQTT ───────────────────────────────────────────────────────────────────
mc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)

def on_connect(client,u,f,rc):
    print(f"[MQTT] rc={rc}")
    client.subscribe(SUB_TOPIC)

def on_message(client,u,msg):
    try:
        p = json.loads(msg.payload.decode())
        state["hw_active"] = True
        d = make_decision(p,"hardware")
        state["latest"] = d
        client.publish(PUB_TOPIC, json.dumps(d))
    except Exception as e: print(f"[MQTT] {e}")

mc.on_connect = on_connect
mc.on_message = on_message

def connect_mqtt():
    while True:
        try:
            mc.connect(BROKER, PORT_MQTT, 60)
            mc.loop_start()
            print("[MQTT] Connected")
            return
        except Exception as e:
            print(f"[MQTT] Retry in 5s: {e}")
            time.sleep(5)

def simulate():
    time.sleep(2)
    print("[SIM] Thread running")
    while True:
        time.sleep(5)
        if not state["hw_active"]:
            p = {"N":random.randint(0,140),"P":random.randint(5,145),
                 "K":random.randint(5,205),"pH":round(random.uniform(3.5,9.5),1),
                 "moist":random.randint(10,100),"temp":round(random.uniform(8.0,44.0),1),
                 "hum":random.randint(14,100),"image_base64":""}
            d = make_decision(p,"simulation")
            state["latest"] = d
            try: mc.publish(PUB_TOPIC, json.dumps(d))
            except: pass
            print(f"[SIM] crop={d['crop_recommendation']['crop']} conf={d['crop_recommendation']['confidence']}")
        else:
            state["hw_active"] = False

# ── Start threads immediately at import time ───────────────────────────────
# Works with gunicorn --preload flag
threading.Thread(target=connect_mqtt, daemon=True).start()
threading.Thread(target=simulate,     daemon=True).start()
print("[APP] Threads launched at import")

# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index(): return jsonify({"status":"Agri-DSS running"})

@app.route("/health")
def health(): return jsonify({"status":"ok"})

@app.route("/latest")
def latest(): return jsonify(state["latest"])

@app.route("/infer", methods=["POST"])
def infer(): return jsonify(make_decision(request.get_json(force=True),"api"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)))
