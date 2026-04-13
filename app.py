import os, json, time, random, threading, io, base64, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

# ── CONFIG ──
STATE_FILE = "/tmp/agri_latest.json"
DISEASE_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy"
]

# ── SMART MODEL LOADING ──
onnx_session = None
MODEL_FILENAME = "disease_resnet18.onnx"

def find_model_path(name):
    """Search for the model file in the current directory and subdirectories."""
    for root, dirs, files in os.walk("."):
        if name in files:
            return os.path.join(root, name)
    return None

try:
    import onnxruntime as ort
    ONNX_PATH = find_model_path(MODEL_FILENAME)
    
    if ONNX_PATH:
        onnx_session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
        print(f"[SUCCESS] Model found and loaded from: {ONNX_PATH}", flush=True)
    else:
        # Final fallback check in root
        if os.path.exists(MODEL_FILENAME):
            onnx_session = ort.InferenceSession(MODEL_FILENAME, providers=["CPUExecutionProvider"])
            print(f"[SUCCESS] Model loaded from root", flush=True)
        else:
            print(f"[ERROR] {MODEL_FILENAME} NOT FOUND ANYWHERE IN REPO", flush=True)
except Exception as e:
    print(f"[ERROR] ONNX Init failed: {e}", flush=True)

# ── IMAGE PROCESSING ──
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def predict_disease(b64_str):
    if onnx_session is None:
        return {"disease": "Model file missing on server", "confidence": 0, "source": "error"}
    try:
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((256, 256))
        
        # Center Crop logic (256 -> 224)
        l, t, r, b = (256-224)/2, (256-224)/2, (256+224)/2, (256+224)/2
        img = img.crop((l, t, r, b))
        
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - IMG_MEAN) / IMG_STD
        arr = arr.transpose(2,0,1)[np.newaxis, ...]
        
        input_name = onnx_session.get_inputs()[0].name
        outputs = onnx_session.run(None, {input_name: arr})[0]
        
        # Softmax & Result
        exp_scores = np.exp(outputs[0] - np.max(outputs[0]))
        probs = exp_scores / exp_scores.sum()
        idx = int(probs.argmax())
        
        return {
            "disease": DISEASE_CLASSES[idx], 
            "confidence": round(float(probs[idx])*100, 2), 
            "source": "model"
        }
    except Exception as e:
        return {"disease": "Inference Error", "confidence": 0, "source": "error"}

# ── ROUTES ──
@app.route("/")
def index():
    return jsonify({
        "status": "running", 
        "model_loaded": onnx_session is not None,
        "search_path": os.getcwd()
    })

@app.route("/latest")
def latest():
    try:
        with open(STATE_FILE) as f: return jsonify(json.load(f))
    except: return jsonify({"error": "No data yet"})

@app.route("/predict_disease", methods=["POST"])
def upload_predict():
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    try:
        img_bytes = request.files['file'].read()
        b64_img = base64.b64encode(img_bytes).decode('utf-8')
        return jsonify(predict_disease(b64_img))
    except Exception as e: return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
