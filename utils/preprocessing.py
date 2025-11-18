from PIL import Image
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input


def preprocess_image(image):
    # Convert BGR → RGB (OpenCV → PIL)
    image = image[:, :, ::-1]

    # Convert to PIL Image
    img = Image.fromarray(image)

    # Resize to EfficientNet input size
    img = img.resize((224, 224))

    # Convert back to numpy array
    img = np.array(img).astype("float32")

    # 🔥 MOST IMPORTANT: EfficientNet normalization
    img = preprocess_input(img)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img
