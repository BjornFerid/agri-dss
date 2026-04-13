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

CROP_CLASSES = [
    "apple","banana","blackgram","chickpea","coconut","coffee","cotton",
    "grapes","jute","kidneybeans","lentil","maize","mango","mothbeans",
    "mungbean","muskmelon","orange","papaya","pigeonpeas","pomegranate",
    "rice","watermelon"
]

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

# -- Preprocessing --
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_image(b64_str):
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((256, 256))
    left, top, right, bottom = (256-224)/2, (256-224)/2, (256+224)/2, (256+224)/2
    img = img.crop((left, top, right, bottom))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMG_MEAN) / IMG_STD
    arr = arr.transpose(2,0,1)
    return arr[np.newaxis, ...].astype(np.float32)

# -- Load Models --
rf_model = None
try:
    for _p in ["crop_model.pkl","crop_rf_model.pkl"]:
        if os.path.exists(_p):
            with open(_p,"rb") as f: rf_model = pickle.load(f)
            break
except: pass

onnx_session = None
# This looks for the file in the SAME folder as app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = os.path.join(BASE_DIR, "disease_resnet18.onnx")

try:
    import onnxruntime as ort
    if os.path.exists(ONNX_PATH):
        # Using CPU only for Render stability
        onnx_session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
        print(f"Model loaded successfully from {ONNX_PATH}")
    else:
        print(f"CRITICAL: {ONNX_PATH} not found!")
except Exception as e:
    print(f"ONNX Error: {e}")

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def predict_disease(b64_str):
    if onnx_session is None:
        return {"disease": "Model Not Loaded", "confidence": 0, "source": "error"}
    try:
        tensor = preprocess_image(b64_str)
        # Verify input name matches your ONNX export (usually 'input' or 'input.1')
        input_name = onnx_session.get_inputs()[0].name
        outputs = onnx_session.run(None, {input_name: tensor})[0]
        probs = softmax(outputs[0])
        idx = int(probs.argmax())
        return {"disease": DISEASE_CLASSES[idx], "confidence": round(float(probs[idx])*100, 2), "source": "model"}
    except Exception as e:
        return {"disease": f"Inference Error", "confidence": 0, "source": "error"}

def write_state(d):
    try:
        with open(STATE_FILE,"w") as f: json.dump(d,f)
    except: pass

def read_state():
    try:
        with open(STATE_FILE) as f: return json.load(f)
    except: return {}

# -- MQTT & Simulator --
def start_mqtt():
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
        client.on_connect = lambda c,u,f,rc: c.subscribe(SUB_TOPIC)
        client.on_message = lambda c,u,msg: write_state(json.loads(msg.payload.decode()))
        client.connect(BROKER,1883,60)
        client.loop_start()
    except: pass

threading.Thread(target=start_mqtt, daemon=True).start()

# -- Routes --
@app.route("/")
def index(): return jsonify({"status":"running"})

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
