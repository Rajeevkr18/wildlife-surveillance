# utils/preprocessing.py

from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.applications.efficientnet import preprocess_input


def preprocess_image(image):
    """
    Preprocess image EXACTLY like during EfficientNet training.
    Accepts RGB or BGR numpy image.
    """

    # --- 1. Ensure RGB ---
    if image.ndim == 3 and image.shape[2] == 3:
        # If input is BGR (OpenCV), convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --- 2. Convert to PIL for resizing ---
    img = Image.fromarray(image.astype("uint8"))

    # --- 3. Resize to EfficientNet input size ---
    img = img.resize((224, 224))

    # --- 4. Convert to numpy ---
    img = np.array(img).astype("float32")

    # --- 5. Apply EfficientNet normalization ---
    img = preprocess_input(img)

    # --- 6. Add batch dimension ---
    img = np.expand_dims(img, axis=0)

    return img
