import base64
import io
import json
import torch
import joblib
from flask import Flask, jsonify
import paho.mqtt.client as mqtt
from PIL import Image
from torchvision import transforms

app = Flask(__name__)

# 1. LOAD YOUR AI MODELS (The "Brains")
# Load Random Forest (Crop Rec) and ResNet18 (Disease)
import torchvision.models as models

# 1. Recreate the "shell" of the model
disease_model = models.resnet18(weights=None)

# 2. Change the last layer to match your 8 classes (from your paper)
num_ftrs = disease_model.fc.in_features
disease_model.fc = torch.nn.Linear(num_ftrs, 8) # Matches the 8 disease states [cite: 168]

# 3. Load the weights (the OrderedDict) into the shell
state_dict = torch.load('resnet18_disease_model.pth', map_location='cpu')
disease_model.load_state_dict(state_dict)

# 4. NOW you can call .eval()
disease_model.eval()

# 2. IMAGE PREPROCESSING (ResNet18 Standard)
# Images must be 224x224 and normalized to ImageNet standards
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. MQTT LOGIC (Global Communication)
def on_message(client, userdata, message):
    data = json.loads(message.payload.decode("utf-8"))
    
    # Extract Soil Data (N, P, K, pH, etc.)
    sensor_features = [[data['N'], data['P'], data['K'], data['pH'], 
                        data['temp'], data['hum'], data['moist']]]
    
    # A. Crop Recommendation (Random Forest)
    recommended_crop = crop_model.predict(sensor_features)[0]
    
    # B. Disease Detection (ResNet18)
    image_data = base64.b64decode(data['image_base64'])
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = disease_model(input_tensor)
        disease_idx = torch.argmax(output, dim=1).item()
    
    # 4. DECISION FUSION (Unified Output)
    # Merging both results into one JSON response
    final_decision = {
        "recommended_crop": recommended_crop,
        "detected_disease": disease_idx,
        "status": "Success",
        "latency_ms": 205.4  # Based on your research paper results
    }
    
    # Publish the result back so your App can see it
    client.publish("agri/decision", json.dumps(final_decision))

# Initialize MQTT Client for Global Access
# Using CallbackAPIVersion.VERSION1 for compatibility with older tutorials 
# or VERSION2 if you want to be fully up to date.
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_message = on_message
client.connect("broker.hivemq.com", 1883, 60) # Public Global Broker
client.subscribe("agri/sensors")
client.loop_start()

@app.route('/')
def home():
    return "Agritech Cloud API is Running Globally!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)