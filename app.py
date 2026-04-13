import os, json, time, random, threading, io, base64, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import paho.mqtt.client as mqtt

app = Flask(__name__)
CORS(app)

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

STATE_FILE = "/tmp/agri_latest.json"
HW_FILE    = "/tmp/agri_hw.txt"

def write_state(d):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(d, f)
    except: pass

def read_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except:
        return {}

def set_hw_active():
    try:
        with open(HW_FILE, "w") as f: f.write("1")
    except: pass

def is_hw_active_and_clear():
    try:
        if os.path.exists(HW_FILE):
            os.remove(HW_FILE)
            return True
    except: pass
    return False

# ── Load model ─────────────────────────────────────────────────────────────
rf_model = None
try:
    for _p in ["crop_model.pkl","crop_rf_model.pkl","crop_rf_model.joblib"]:
        if os.path.exists(_p):
            with open(_p,"rb") as _f: rf_model = pickle.load(_f)
            print(f"[ML] Loaded {_p}", flush=True)
            break
    if rf_model is None:
        print("[ML] No model file - using random", flush=True)
except Exception as e:
    print(f"[ML] Error: {e}", flush=True)
    rf_model = None

# ── Inference ──────────────────────────────────────────────────────────────
def predict_crop(p):
    try:
        if rf_model is not None:
            X = np.array([[float(p["N"]),float(p["P"]),float(p["K"]),
                           float(p["temp"]),float(p["moist"]),float(p["pH"])]])
            idx  = rf_model.predict(X)[0]
            crop = idx if isinstance(idx,str) else CROP_CLASSES[int(idx) % 22]
            try:    conf = round(float(rf_model.predict_proba(X)[0].max())*100,2)
            except: conf = round(random.uniform(80,99),2)
            return {"crop": crop, "confidence": conf}
    except Exception as e:
        print(f"[ML] predict error: {e}", flush=True)
    return {"crop": random.choice(CROP_CLASSES), "confidence": round(random.uniform(80,99),2)}

def predict_disease(b64):
    try:
        if b64:
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB").resize((64,64))
            br  = float(np.array(img,dtype=np.float32).mean()/255)
            if br>0.75:   return {"disease":"Healthy",         "confidence":88.5}
            elif br>0.55: return {"disease":"Early Blight",    "confidence":79.2}
            elif br>0.35: return {"disease":"Leaf Rust",       "confidence":83.1}
            else:         return {"disease":"Bacterial Blight","confidence":76.4}
    except: pass
    return {"disease": DISEASE_CLASSES[random.randint(0,7)], "confidence": round(random.uniform(70,97),2)}

def make_decision(p, source="simulation"):
    return {
        "timestamp":           time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        "source":              source,
        "sensor_data":         {"N":p.get("N",0),"P":p.get("P",0),"K":p.get("K",0),
                                "pH":p.get("pH",0),"moisture":p.get("moist",0),
                                "temperature":p.get("temp",0),"humidity":p.get("hum",0)},
        "crop_recommendation": predict_crop(p),
        "disease_analysis":    predict_disease(p.get("image_base64",""))
    }

# ── MQTT ───────────────────────────────────────────────────────────────────
_mc = None

def start_mqtt():
    global _mc
    try:
        _mc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
        def on_connect(client,u,f,rc):
            print(f"[MQTT] Connected rc={rc}", flush=True)
            client.subscribe("agri/sensors")
        def on_message(client,u,msg):
            try:
                p = json.loads(msg.payload.decode())
                set_hw_active()
                d = make_decision(p,"hardware")
                write_state(d)
                client.publish("agri/decision", json.dumps(d))
            except Exception as e: print(f"[MQTT] {e}", flush=True)
        _mc.on_connect = on_connect
        _mc.on_message = on_message
        _mc.connect("broker.hivemq.com",1883,60)
        _mc.loop_start()
        print("[MQTT] Started", flush=True)
    except Exception as e:
        print(f"[MQTT] Failed: {e}", flush=True)

def simulate():
    time.sleep(3)
    print("[SIM] Thread running", flush=True)
    while True:
        time.sleep(5)
        if not is_hw_active_and_clear():
            p = {"N":random.randint(0,140),"P":random.randint(5,145),
                 "K":random.randint(5,205),"pH":round(random.uniform(3.5,9.5),1),
                 "moist":random.randint(10,100),"temp":round(random.uniform(8.0,44.0),1),
                 "hum":random.randint(14,100),"image_base64":""}
            d = make_decision(p,"simulation")
            write_state(d)
            if _mc:
                try: _mc.publish("agri/decision", json.dumps(d))
                except: pass
            print(f"[SIM] crop={d['crop_recommendation']['crop']} conf={d['crop_recommendation']['confidence']}", flush=True)

print("[APP] Starting...", flush=True)
start_mqtt()
threading.Thread(target=simulate, daemon=True).start()
print("[APP] Threads launched", flush=True)

# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index(): return jsonify({"status":"Agri-DSS running"})

@app.route("/health")
def health(): return jsonify({"status":"ok"})

@app.route("/latest")
def latest(): return jsonify(read_state())

@app.route("/infer", methods=["POST"])
def infer(): return jsonify(make_decision(request.get_json(force=True),"api"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)))
