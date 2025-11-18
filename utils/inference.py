import tensorflow as tf
import numpy as np
import cv2
import os
import json
import gdown
from utils.preprocessing import preprocess_image

# ---------------------------
# Resolve absolute project path
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# ---------------------------
# Google Drive model file ID
# ---------------------------
MODEL_FILE_ID = "1tV1RDM4ZuxNmSHnxl_xWh3DMu6ZhHzRi"

# Local path where model should be stored
MODEL_PATH = os.path.join(BASE_DIR, "models", "animal_classifier.keras")

# ---------------------------
# Download model if NOT present
# ---------------------------
if not os.path.exists(MODEL_PATH):
    print("Model not found. Downloading from Google Drive...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# ---------------------------
# Load model
# ---------------------------
animal_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully!")

# ---------------------------
# Load class mapping
# ---------------------------
json_path = os.path.join(BASE_DIR, "class_to_idx.json")

with open(json_path, "r") as f:
    class_map = json.load(f)

idx_to_class = {int(v): k for k, v in class_map.items()}


# ---------------------------------------------------
# ANIMAL PREDICTION
# ---------------------------------------------------
def predict_animal(image):
    img = preprocess_image(image)
    preds = animal_model.predict(img)[0]

    idx = int(np.argmax(preds))
    label = idx_to_class[idx]
    confidence = float(preds[idx])

    # FIX: Return lowercase for Wikipedia lookup
    return label.lower(), confidence


# ---------------------------------------------------
# FIRE PREDICTION (DUMMY)
# ---------------------------------------------------
def predict_fire(image):
    return "No Fire", 0.98


# ---------------------------------------------------
# POACHING DETECTION (FIXED RECTANGLE)
# ---------------------------------------------------
def detect_poaching(image):
    h, w, _ = image.shape

    # FIXED OpenCV rectangle → MUST use tuple (w-50, h-50)
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
