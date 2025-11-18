# utils/inference.py
import sys
import os

# Ensure utils directory is on sys.path (works both locally & on Streamlit Cloud)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import tensorflow as tf
import numpy as np
import cv2
import json
import gdown
from preprocessing import preprocess_image   # local import (preprocessing.py in utils/)

# ---------------------------
# BASE / MODEL PATHS
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))   # project root
MODEL_FILE_ID = "1tV1RDM4ZuxNmSHnxl_xWh3DMu6ZhHzRi"
MODEL_PATH = os.path.join(BASE_DIR, "models", "animal_classifier.keras")

# ---------------------------
# DOWNLOAD MODEL IF MISSING
# ---------------------------
if not os.path.exists(MODEL_PATH):
    print("Model not found. Downloading from Google Drive...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", MODEL_PATH, quiet=False)

# ---------------------------
# LOAD MODEL
# ---------------------------
animal_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully!")

# ---------------------------
# LOAD CLASS MAPPING
# ---------------------------
json_path = os.path.join(BASE_DIR, "class_to_idx.json")
with open(json_path, "r") as f:
    class_map = json.load(f)
idx_to_class = {int(v): k for k, v in class_map.items()}

# ---------------------------
# PREDICTION FUNCTIONS
# ---------------------------
def predict_animal(image):
    """image: numpy array RGB"""
    img = preprocess_image(image)                 # returns batch shape (1,224,224,3)
    preds = animal_model.predict(img)[0]
    idx = int(np.argmax(preds))
    label = idx_to_class[idx]
    confidence = float(preds[idx])
    return label, confidence

def predict_fire(image):
    return "No Fire", 0.98

def detect_poaching(image):
    h, w, _ = image.shape
    det_img = cv2.rectangle(image.copy(), (50, 50), (w - 50, h - 50), (0, 255, 0), 3)
    class Dummy:
        def __init__(self):
            self.xyxy = np.array([[50, 50, w - 50, h - 50]])
    return det_img, [Dummy()]
