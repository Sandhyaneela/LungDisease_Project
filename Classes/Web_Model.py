import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
import os

MODEL_PATH = 'runs/best_weight.h5'
CLASS_INDICES_PATH = 'Classes/class_indices.json'

model = load_model(MODEL_PATH)

with open(CLASS_INDICES_PATH) as f:
    class_indices = json.load(f)

# Invert class indices
class_names = {v: k for k, v in class_indices.items()}

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    predicted_index = np.argmax(predictions)
    confidence = predictions[predicted_index] * 100
    predicted_class = class_names[predicted_index]

    return predicted_class, confidence
