from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.image import img_to_array, load_img 
from PIL import Image
import numpy as np
import os
import json
import pandas as pd

app = Flask(__name__)
CORS(app)

# Directory for uploaded files
UPLOAD_FOLDER = './../uploads'
UPLOAD_FOLDER = os.path.abspath(UPLOAD_FOLDER)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model("./best_pest_detection_model.keras")
print('Model Loaded successfully')

CLASS_INDICES_PATH = './../output/class_indices.json'

try:
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    class_indices = {v: k for k, v in class_indices.items()}  # Reverse mapping
    print("Class indices loaded successfully!")
except FileNotFoundError:
    raise FileNotFoundError(f"Class indices file not found at {CLASS_INDICES_PATH}")


# Load Pesticide Data
PESTICIDE_DATA_PATH = './Pesticide data set.xlsx'
pesticide_data = pd.read_excel(PESTICIDE_DATA_PATH)
pesticide_data.columns = pesticide_data.columns.str.strip()
print("Pesticide data loaded successfully!")

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_pesticide(image_path, model, pesticide_data, class_indices):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    pest_name = class_indices[predicted_class]

    recommendation = pesticide_data[pesticide_data['Pest'].str.contains(pest_name, case=False, na=False)]
    if not recommendation.empty:
        pesticide = recommendation.iloc[0]['Pesticide']
        alternative = recommendation.iloc[0]['Alternative']
    else:
        pesticide = "No recommendation found"
        alternative = "No alternative found"

    return pest_name, pesticide, alternative

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Pest Detection and Pesticide prediction"})

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    print(filepath)
    file.save(filepath)
    return jsonify({"message": f"File {file.filename} uploaded successfully!", "path": filepath})

@app.route("/predict-image", methods=["POST"])
def predict_image():
    data = request.get_json()
    file_path = data.get('file_path')

    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 400

    try:
        pest, pesticide, alternative = predict_pesticide(file_path, model, pesticide_data, class_indices)
        response = f"Pest Name: {pest}\nPesticide: {pesticide}\nAlternative: {alternative}"
        return response, 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
