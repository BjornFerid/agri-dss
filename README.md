# Agri-Tech DSS — Hybrid AI Decision Support System

## Architecture

```
ESP32-CAM  ──MQTT──▶  broker.hivemq.com  ──MQTT──▶  Flask API (Render)
                                                          │
                                                    RF + ResNet-18
                                                          │
                              Dashboard  ◀──MQTT──  agri/decision
```

## Files

| File | Purpose |
|---|---|
| `app.py` | Flask API — MQTT subscriber, ML inference, simulation thread |
| `requirements.txt` | Python dependencies |
| `render.yaml` | Render PaaS deployment config |
| `esp32_edge_node.ino` | Arduino sketch for ESP32-CAM edge node |
| `dashboard.html` | HTML5 real-time dashboard (MQTT over WebSocket) |

---

## 1. Train & Export Your Models

### Random Forest (crop recommendation)
```python
from sklearn.ensemble import RandomForestClassifier
import joblib, pandas as pd

df  = pd.read_csv("crop_data.csv")   # N,P,K,pH,moist,temp,hum,label
X   = df[["N","P","K","pH","moist","temp","hum"]]
y   = df["label"]
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)
joblib.dump(clf, "crop_rf_model.joblib")
```

### ResNet-18 (disease detection)
```python
import torch, torchvision.models as models

model    = models.resnet18(weights="DEFAULT")
model.fc = torch.nn.Linear(model.fc.in_features, 8)
# ... fine-tune on your foliage dataset ...
torch.save(model.state_dict(), "disease_resnet18.pth")
```

Place both files next to `app.py` before deploying.

---

## 2. Deploy Backend to Render

1. Push this repo to GitHub.
2. Create a new **Web Service** on [render.com](https://render.com).
3. Connect your GitHub repo — Render auto-detects `render.yaml`.
4. Set environment variable `PORT=5000` if not already in `render.yaml`.
5. Deploy. Your API URL will be `https://<your-app>.onrender.com`.

### API endpoints
| Method | Route | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/latest` | Last inference result as JSON |
| POST | `/infer` | Manual inference (send JSON payload) |

---

## 3. Flash ESP32-CAM

1. Open `esp32_edge_node.ino` in Arduino IDE.
2. Install libraries: **PubSubClient**, **ArduinoJson**, **ESP32 Camera**.
3. Set `WIFI_SSID` and `WIFI_PASSWORD`.
4. Select board: **AI Thinker ESP32-CAM**, upload.
5. Replace stub sensor functions with real sensor library calls.

> **Note:** The MQTT buffer is set to 64 KB to accommodate Base64-encoded JPEG images. Increase `setBufferSize()` if images are larger.

---

## 4. Open the Dashboard

Open `dashboard.html` in any browser — no server needed.  
It connects directly to `broker.hivemq.com` over WebSocket (WSS port 8884).

The **state-lock** (`hasData`) prevents UI flickering:
- First packet → renders immediately.
- Subsequent packets → queued; a refresh banner appears.
- Clicking "Refresh" reloads the page with the latest data.

---

## 5. Simulation Mode

When no ESP32 hardware is publishing, `app.py` automatically generates random sensor values every **5 seconds** and runs full inference. The dashboard marks these with a `Simulated` badge. Simulation stops automatically when real hardware begins publishing.

---

## MQTT Topics

| Topic | Direction | Content |
|---|---|---|
| `agri/sensors` | ESP32 → Cloud | Raw sensor + Base64 image JSON |
| `agri/decision` | Cloud → Dashboard | Inference results JSON |

### Decision payload schema
```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "source": "hardware | simulation",
  "sensor_data": { "N": 80, "P": 40, "K": 60, "pH": 6.5,
                   "moisture": 65, "temperature": 28.5, "humidity": 72 },
  "crop_recommendation": { "crop": "rice", "confidence": 94.2 },
  "disease_analysis":    { "disease": "Healthy", "confidence": 91.7 }
}
```
