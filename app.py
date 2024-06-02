from flask import Flask, request, jsonify
import face_recognition
import requests
from io import BytesIO
import base64
import cv2
import numpy as np

app = Flask(__name__)

def convert_and_upscale_image(byte):
    if isinstance(byte, str):
        # Decode base64 data into image bytes
        byte = byte.split(",")[1]
        image_bytes = base64.b64decode(byte)

        # Convert image bytes to numpy array
        image_array = np.frombuffer(image_bytes, np.uint8)

        # Decode the numpy array into an image
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Get original dimensions
        height, width, _ = image.shape

        # Define the upscale factor (e.g., 2x)
        upscale_factor = 4

        # Calculate new dimensions
        new_height = height * upscale_factor
        new_width = width * upscale_factor

        # Upscale the image
        upscaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    else:
        upscaled_image = face_recognition.load_image_file(BytesIO(byte))

    return upscaled_image

@app.route('/face_recognition', methods=['POST'])
def face_recognition_endpoint():
    try:
        image1 = request.json['image1']
        image2 = request.json['image2']

        if not image1 or not image2:
            return jsonify({"error": "Please provide both image URLs"}), 400


        upscaled_image1 = convert_and_upscale_image(image1)
        upscaled_image2 = convert_and_upscale_image(image2)

        face_recognition_face_encoding1 = face_recognition.face_encodings(upscaled_image1)[0]
        face_recognition_face_encoding2 = face_recognition.face_encodings(upscaled_image2)[0]

        distance = face_recognition.face_distance([face_recognition_face_encoding1], face_recognition_face_encoding2)[0]
        distance = round(100 - distance * 100, 2)

        if distance >= 52:
            result = "Match"
        else:
            result = "No Match"

        return jsonify({"distance": distance, "result": result})

    except Exception as e:
        return jsonify({"error": "Invalid Image"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)