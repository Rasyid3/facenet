from flask import Flask, request, jsonify
import torch
import requests
from torchvision import transforms
from PIL import Image
import io
import json
from facenet_pytorch import MTCNN, InceptionResnetV1
import logging
import numpy as np
from scipy.spatial.distance import cosine
from flask_cors import CORS
import threading
import time

ESP32_CAM_URL = "http://192.168.127.196/capture"
ESP32_RELAY_URL = "http://192.168.127.1/relay?state=ON"
LARAVEL_BACKEND_URL = "https://trahmartorejan.site/api/wajah"
API_URL = "https://trahmartorejan.site/api/wajah"
CHECK_INTERVAL = 5  # seconds

app = Flask(__name__)
CORS(app)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)


def get_all_embeddings():
    try:
        logger.info(f"Fetching all embeddings from {API_URL}")
        response = requests.get(API_URL)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                return [(entry["embedding"], entry["wajahable_type"], entry["wajahable_id"]) for entry in data]
        logger.error("Unexpected API response format")
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
    return []


def compare_with_database(input_embedding, threshold=0.8):
    stored_data = get_all_embeddings()
    for stored_embedding, wajahable_type, wajahable_id in stored_data:
        stored_embedding = np.array(stored_embedding)
        similarity = 1 - cosine(input_embedding, stored_embedding)
        logger.info(f"Comparing with {wajahable_type} (ID: {wajahable_id}) - Similarity: {similarity:.4f}")
        if similarity >= threshold:
            logger.info(f"Match found with {wajahable_type} (ID: {wajahable_id}) at similarity {similarity:.4f}")
            return True
    return False


def preprocess_image(image_data):
    img = Image.open(io.BytesIO(image_data))
    faces = mtcnn(img)
    if faces is None:
        raise ValueError("No face detected.")
    return faces


def image_to_embedding(faces):
    with torch.no_grad():
        if faces.dim() == 5:
            faces = faces.squeeze(0)
        return resnet(faces).squeeze().numpy().tolist()


@app.route('/process-image/', methods=['POST'])
def process_image():
    try:
        image = request.files.get('image')
        wajahable_type = request.form.get('wajahable_type')
        wajahable_id = request.form.get('wajahable_id')

        if not image or not wajahable_type or not wajahable_id:
            return jsonify({"error": "Missing required fields"}), 400

        image_bytes = image.read()
        faces = preprocess_image(image_bytes)
        embedding = image_to_embedding(faces)

        data = {
            "wajahable_type": wajahable_type.lower(),
            "wajahable_id": int(wajahable_id),
            "embedding": embedding
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(LARAVEL_BACKEND_URL, json=data, headers=headers)

        return jsonify(response.json())

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Run Flask app
    app.run(debug=True, host='0.0.0.0')
