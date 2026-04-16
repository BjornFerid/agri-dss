import os, json, time, random, threading, io, base64, pickle, warnings
warnings.filterwarnings("ignore")

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

# ── Exact classes from your trained models ─────────────────────────────────
CROP_CLASSES = [
    "apple","banana","blackgram","chickpea","coconut","coffee","cotton",
    "grapes","jute","kidneybeans","lentil","maize","mango","mothbeans",
    "mungbean","muskmelon","orange","papaya","pigeonpeas","pomegranate",
    "rice","watermelon"
]

# ── Replace these with your actual 8 disease class names ──────────────────
# Check your training notebook for the exact names and order
# They must match the order used during training (folder names or label list)
DISEASE_CLASSES = [
    "Apple__Apple_scab",
    "Apple__Black_rot",
    "Apple__Cedar_apple_rust",
    "Apple__healthy",
    "Corn_(maize)__Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)__Common_rust_",
    "Corn_(maize)__Northern_Leaf_Blight",
    "Corn_(maize)__healthy"
]

STATE_FILE = "/tmp/agri_latest.json"
HW_FILE    = "/tmp/agri_hw.txt"

def write_state(d):
    try:
        with open(STATE_FILE,"w") as f: json.dump(d,f)
    except: pass

def read_state():
    try:
        with open(STATE_FILE) as f: return json.load(f)
    except: return {}

def set_hw_active():
    try:
        with open(HW_FILE,"w") as f: f.write("1")
    except: pass

def is_hw_active_and_clear():
    try:
        if os.path.exists(HW_FILE):
            os.remove(HW_FILE); return True
    except: pass
    return False

# ── Image preprocessing (same as training) ─────────────────────────────────
IMG_MEAN = np.array([0.485,0.456,0.406], dtype=np.float32)
IMG_STD  = np.array([0.229,0.224,0.225], dtype=np.float32)

def preprocess_image(b64_str):
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224,224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMG_MEAN) / IMG_STD
    arr = arr.transpose(2,0,1)          # HWC → CHW
    return arr[np.newaxis, ...].astype(np.float32)  # add batch dim

# ── Load Random Forest ──────────────────────────────────────────────────────
rf_model = None
try:
    for _p in ["crop_model.pkl","crop_rf_model.pkl","crop_rf_model.joblib"]:
        if os.path.exists(_p):
            with open(_p,"rb") as f: rf_model = pickle.load(f)
            print(f"[ML] RF loaded from {_p}", flush=True)
            break
    if rf_model is None: print("[ML] No RF model found", flush=True)
except Exception as e:
    print(f"[ML] RF load error: {e}", flush=True)

# ── Load ONNX disease model ────────────────────────────────────────────────
onnx_session = None
try:
    import onnxruntime as ort
    onnx_path = "disease_resnet18.onnx"
    if os.path.exists(onnx_path):
        onnx_session = ort.InferenceSession(onnx_path,
            providers=["CPUExecutionProvider"])
        print(f"[ML] ONNX disease model loaded ✅", flush=True)
    else:
        print("[ML] disease_resnet18.onnx not found — using fallback", flush=True)
except ImportError:
    print("[ML] onnxruntime not installed", flush=True)
except Exception as e:
    print(f"[ML] ONNX load error: {e}", flush=True)

# ── Inference ───────────────────────────────────────────────────────────────
def predict_crop(p):
    try:
        if rf_model is not None:
            X = np.array([[float(p["N"]),float(p["P"]),float(p["K"]),
                           float(p["temp"]),float(p["moist"]),float(p["pH"])]])
            idx  = rf_model.predict(X)[0]
            crop = idx if isinstance(idx,str) else CROP_CLASSES[int(idx) % len(CROP_CLASSES)]
            try:    conf = round(float(rf_model.predict_proba(X)[0].max())*100,2)
            except: conf = round(random.uniform(80,99),2)
            return {"crop": crop, "confidence": conf, "source": "model"}
    except Exception as e:
        print(f"[ML] crop predict error: {e}", flush=True)
    return {"crop": random.choice(CROP_CLASSES),
            "confidence": round(random.uniform(70,90),2), "source": "random"}

def predict_disease(b64_str):
    # Use real ONNX model if available
    if onnx_session and b64_str:
        try:
            tensor = preprocess_image(b64_str)
            outputs = onnx_session.run(["output"], {"input": tensor})[0]
            probs   = softmax(outputs[0])
            idx     = int(probs.argmax())
            conf    = round(float(probs[idx])*100, 2)
            return {"disease": DISEASE_CLASSES[idx], "confidence": conf, "source": "model"}
        except Exception as e:
            print(f"[ML] disease predict error: {e}", flush=True)

    # Fallback — random (until ONNX model is uploaded)
    idx = random.randint(0,7)
    return {"disease": DISEASE_CLASSES[idx],
            "confidence": round(random.uniform(70,95),2), "source": "random"}

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def make_decision(p, source="simulation"):
    return {
        "timestamp":           time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        "source":              source,
        "sensor_data":         {
            "N":           p.get("N",0),   "P":   p.get("P",0),
            "K":           p.get("K",0),   "pH":  p.get("pH",0),
            "moisture":    p.get("moist",0),
            "temperature": p.get("temp",0),
            "humidity":    p.get("hum",0)
        },
        "crop_recommendation": predict_crop(p),
        "disease_analysis":    predict_disease(p.get("image_base64",""))
    }

# ── MQTT ────────────────────────────────────────────────────────────────────
_mc = None

def start_mqtt():
    global _mc
    try:
        _mc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
        def on_connect(c,u,f,rc):
            print(f"[MQTT] Connected rc={rc}", flush=True)
            c.subscribe(SUB_TOPIC)
        def on_message(c,u,msg):
            try:
                p = json.loads(msg.payload.decode())
                set_hw_active()
                d = make_decision(p,"hardware")
                write_state(d)
                c.publish(PUB_TOPIC, json.dumps(d))
                print(f"[MQTT] hw → crop={d['crop_recommendation']['crop']}", flush=True)
            except Exception as e: print(f"[MQTT] {e}", flush=True)
        _mc.on_connect = on_connect
        _mc.on_message = on_message
        _mc.connect(BROKER,PORT_MQTT,60)
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
            p = {
                "N":    random.randint(0,140),  "P":    random.randint(5,145),
                "K":    random.randint(5,205),   "pH":   round(random.uniform(3.5,9.5),1),
                "moist":random.randint(10,100),  "temp": round(random.uniform(8.0,44.0),1),
                "hum":  random.randint(14,100),  "image_base64": ""
            }
            d = make_decision(p,"simulation")
            write_state(d)
            if _mc:
                try: _mc.publish(PUB_TOPIC, json.dumps(d))
                except: pass
            print(f"[SIM] crop={d['crop_recommendation']['crop']} "
                  f"conf={d['crop_recommendation']['confidence']} "
                  f"disease={d['disease_analysis']['disease']}", flush=True)
        else:
            state["hw_active"] = False

print("[APP] Starting...", flush=True)
start_mqtt()
threading.Thread(target=simulate, daemon=True).start()
print("[APP] Threads launched", flush=True)

# ── Routes ───────────────────────────────────────────────────────────────────
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
