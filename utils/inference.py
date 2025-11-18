import sys
import os

# ==========================================================
# FIX PATH FOR STREAMLIT CLOUD TO IMPORT preprocessing.py
# ==========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import tensorflow as tf
import numpy as np
import cv2
import json
import gdown
from preprocessing import preprocess_image    # <-- FIXED IMPORT


# ==========================================================
# BASE PROJECT PATH
# ==========================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Google Drive ID of your model
MODEL_FILE_ID = "1tV1RDM4ZuxNmSHnxl_xWh3DMu6ZhHzRi"

# Model storage path
MODEL_PATH = os.path.join(BASE_DIR, "models", "animal_classifier.keras")


# ==========================================================
# DOWNLOAD MODEL IF NOT PRESENT
# ==========================================================
if not os.path.exists(MODEL_PATH):
    print("Model not found. Downloading from Google Drive...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    gdown.download(
        f"https://drive.google.com/uc?id={MODEL_FILE_ID}",
        MODEL_PATH,
        quiet=False
    )


# ==========================================================
# LOAD MODEL
# ==========================================================
animal_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully!")


# ==========================================================
# LOAD CLASS MAPPING (class_to_idx.json)
# ==========================================================
json_path = os.path.join(BASE_DIR, "class_to_idx.json")

with open(json_path, "r") as f:
    class_map = json.load(f)

idx_to_class = {int(v): k for k, v in class_map.items()}


# ==========================================================
# ANIMAL PREDICTION
# ==========================================================
def predict_animal(image):
    img = preprocess_image(image)
    preds = animal_model.predict(img)[0]

    idx = int(np.argmax(preds))
    label = idx_to_class[idx].lower()   # ALWAYS lower-case
    confidence = float(preds[idx])

    return label, confidence


# ==========================================================
# FIRE (DUMMY DETECTOR)
# ==========================================================
def predict_fire(image):
    return "No Fire", 0.98


# ==========================================================
# POACHING DETECTION (RECTANGLE FIXED)
# ==========================================================
def detect_poaching(image):
    h, w, _ = image.shape

    det_img = cv2.rectangle(
        image.copy(),
        (50, 50),
        (w - 50, h - 50),
        (0, 255, 0),
        3
    )

    class Dummy:
        def __init__(self):
            self.xyxy = np.array([[50, 50, w - 50, h - 50]])

    return det_img, [Dummy()]
