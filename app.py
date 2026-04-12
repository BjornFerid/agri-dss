import os, json, builtins
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return jsonify({"status": "Agri-DSS API running"})

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/latest")
def latest():
    return jsonify(getattr(builtins, "_agri_latest", {}))

@app.route("/infer", methods=["POST"])
def infer():
    # inline import so gunicorn_config state is available
    from gunicorn_config import run_inference  # noqa
    return jsonify(run_inference(request.get_json(force=True)))

if __name__ == "__main__":
    # Dev mode — start manually
    import threading, time, random, pickle, numpy as np, io, base64
    from PIL import Image
    import paho.mqtt.client as mqtt

    print("[DEV] Starting in dev mode...")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
