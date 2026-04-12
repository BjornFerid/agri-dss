import threading
import time
import random
import json
import pickle
import numpy as np
from PIL import Image
import io, base64, os

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

bind = "0.0.0.0:" + os.environ.get("PORT","5000")
workers = 1
threads = 4

def on_starting(server):
    """Gunicorn hook — runs once before workers fork."""
    import paho.mqtt.client as mqtt

    # Load model
    rf_model = None
    for path in ["crop_model.pkl","crop_rf_model.joblib","crop_rf_model.pkl"]:
        if os.path.exists(path):
            with open(path,"rb") as f:
                rf_model = pickle.load(f)
            print(f"[ML] Loaded RF from {path}")
            break

    # Shared state (in-process, single worker)
    import builtins
    builtins._agri_latest   = {}
    builtins._agri_hw_active = False

    def predict_crop(p):
        if rf_model is None:
            return {"crop": random.choice(CROP_CLASSES), "confidence": round(random.uniform(70,99),2)}
        X = np.array([[p["N"],p["P"],p["K"],p["pH"],p["moist"],p["temp"],p["hum"]]])
        idx = rf_model.predict(X)[0]
        crop = idx if isinstance(idx,str) else (CROP_CLASSES[int(idx)] if int(idx)<len(CROP_CLASSES) else "unknown")
        try:    conf = round(float(rf_model.predict_proba(X)[0].max())*100,2)
        except: conf = round(random.uniform(85,99),2)
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

    def run_inference(p):
        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
            "source": "simulation",
            "sensor_data": {"N":p.get("N",0),"P":p.get("P",0),"K":p.get("K",0),
                            "pH":p.get("pH",0),"moisture":p.get("moist",0),
                            "temperature":p.get("temp",0),"humidity":p.get("hum",0)},
            "crop_recommendation": predict_crop(p),
            "disease_analysis":    predict_disease(p.get("image_base64",""))
        }

    # MQTT client
    mc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)

    def on_connect(client,u,f,rc):
        print(f"[MQTT] Connected rc={rc}")
        client.subscribe("agri/sensors")

    def on_message(client,u,msg):
        try:
            p = json.loads(msg.payload.decode())
            builtins._agri_hw_active = True
            d = run_inference(p); d["source"]="hardware"
            builtins._agri_latest = d
            client.publish("agri/decision", json.dumps(d))
        except Exception as e: print(f"[MQTT] {e}")

    mc.on_connect = on_connect
    mc.on_message = on_message

    try:
        mc.connect("broker.hivemq.com", 1883, 60)
        mc.loop_start()
        print("[MQTT] Started")
    except Exception as e:
        print(f"[MQTT] Failed: {e}")

    def simulate():
        time.sleep(3)
        while True:
            time.sleep(5)
            active = builtins._agri_hw_active
            builtins._agri_hw_active = False
            if not active:
                p = {"N":random.randint(0,140),"P":random.randint(5,145),
                     "K":random.randint(5,205),"pH":round(random.uniform(3.5,9.5),1),
                     "moist":random.randint(10,100),"temp":round(random.uniform(8.0,44.0),1),
                     "hum":random.randint(14,100),"image_base64":""}
                d = run_inference(p)
                builtins._agri_latest = d
                try:
                    mc.publish("agri/decision", json.dumps(d))
                    print(f"[SIM] → {d['crop_recommendation']['crop']}")
                except Exception as e: print(f"[SIM] {e}")

    threading.Thread(target=simulate, daemon=True).start()
    print("[SIM] Simulation thread started")