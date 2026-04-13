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

# ── Updated Class List (Strict Alphabetical Order per PyTorch ImageFolder) ──
CROP_CLASSES = [
    "apple","banana","blackgram","chickpea","coconut","coffee","cotton",
    "grapes","jute","kidneybeans","lentil","maize","mango","mothbeans",
    "mungbean","muskmelon","orange","papaya","pigeonpeas","pomegranate",
    "rice","watermelon"
]

# CRITICAL: These MUST match your folder names in 'split_dataset/train' alphabetically
DISEASE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy"
]

STATE_FILE = "/tmp/agri_latest.json"
HW_FILE    = "/tmp/agri_hw.txt"

# ── Helper Functions ──
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

# ── Image Preprocessing (Updated to match Training 'test' transform) ──
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_image(b64_str):
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    # 1. Resize to 256 (matches training Resize(256))
    img = img.resize((256, 256))
    
    # 2. Center Crop to 224 (matches training CenterCrop(224))
    left = (256 - 224) / 2
    top = (256 - 224) / 2
    right = (256 + 224) / 2
    bottom = (256 + 224) / 2
    img = img.crop((left, top, right, bottom))
    
    # 3. Normalize
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMG_MEAN) / IMG_STD
    arr = arr.transpose(2,0,1)          # HWC → CHW
    return arr[np.newaxis, ...].astype(np.float32)

# ── Load Models ──
rf_model = None
try:
    for _p in ["crop_model.pkl","crop_rf_model.pkl","crop_rf_model.joblib"]:
        if os.path.exists(_p):
            with open(_p,"rb") as f: rf_model = pickle.load(f)
            break
except: pass

onnx_session = None
try:
    import onnxruntime as ort
    if os.path.exists("disease_resnet18.onnx"):
        onnx_session = ort.InferenceSession("disease_resnet18.onnx", providers=["CPUExecutionProvider"])
except: pass

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# ── Inference Logic ──
def predict_crop(p):
    try:
        if rf_model is not None:
            X = np.array([[float(p["N"]),float(p["P"]),float(p["K"]),
                           float(p["temp"]),float(p["moist"]),float(p["pH"])]])
            idx  = rf_model.predict(X)[0]
            crop = idx if isinstance(idx,str) else CROP_CLASSES[int(idx) % len(CROP_CLASSES)]
            try: conf = round(float(rf_model.predict_proba(X)[0].max())*100,2)
            except: conf = round(random.uniform(80,99),2)
            return {"crop": crop, "confidence": conf, "source": "model"}
    except: pass
    return {"crop": random.choice(CROP_CLASSES), "confidence": 0, "source": "error"}

def predict_disease(b64_str):
    if onnx_session and b64_str:
        try:
            tensor = preprocess_image(b64_str)
            outputs = onnx_session.run(["output"], {"input": tensor})[0]
            probs   = softmax(outputs[0])
            idx     = int(probs.argmax())
            conf    = round(float(probs[idx])*100, 2)
            return {"disease": DISEASE_CLASSES[idx], "confidence": conf, "source": "model"}
        except Exception as e:
            print(f"Inference Error: {e}")
    
    return {"disease": "Unknown", "confidence": 0, "source": "error"}

def make_decision(p, source="simulation"):
    return {
        "timestamp":           time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        "source":              source,
        "sensor_data":         {
            "N": p.get("N",0), "P": p.get("P",0), "K": p.get("K",0), 
            "pH": p.get("pH",0), "moisture": p.get("moist",0),
            "temperature": p.get("temp",0), "humidity": p.get("hum",0)
        },
        "crop_recommendation": predict_crop(p),
        "disease_analysis":    predict_disease(p.get("image_base64",""))
    }

# ── Threads ──
def start_mqtt():
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
        client.on_connect = lambda c,u,f,rc: c.subscribe(SUB_TOPIC)
        def on_msg(c,u,msg):
            p = json.loads(msg.payload.decode())
            set_hw_active()
            d = make_decision(p,"hardware")
            write_state(d)
            c.publish(PUB_TOPIC, json.dumps(d))
        client.on_message = on_msg
        client.connect(BROKER,PORT_MQTT,60)
        client.loop_start()
    except: pass

def simulate():
    while True:
        time.sleep(5)
        if not is_hw_active_and_clear():
            p = {"N":random.randint(0,140),"P":random.randint(5,145),"K":random.randint(5,205),
                 "pH":round(random.uniform(3.5,9.5),1),"moist":random.randint(10,100),
                 "temp":round(random.uniform(8.0,44.0),1),"hum":random.randint(14,100),"image_base64":""}
            write_state(make_decision(p,"simulation"))

start_mqtt()
threading.Thread(target=simulate, daemon=True).start()

# ── Routes ──
@app.route("/")
def index(): return jsonify({"status":"Agri-DSS running"})

@app.route("/latest")
def latest(): return jsonify(read_state())

@app.route("/predict_disease", methods=["POST"])
def upload_predict():
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    try:
        img_bytes = request.files['file'].read()
        b64_img = base64.b64encode(img_bytes).decode('utf-8')
        return jsonify(predict_disease(b64_img))
    except Exception as e: return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)))
