from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from flask_cors import CORS
import base64
import io

app = Flask(__name__)
CORS(app)  # To handle cross-origin requests

# Load the trained model
model = tf.keras.models.load_model('mobilenetv2_skin_tone.h5')  # Update with your model file path
input_shape = (90, 120)  # Height, Width (expected by the model)

# Map predicted classes to skin tone labels
skin_tone_labels = {
    0: "Black",
    1: "Brwon",
    2: "White"
}

# Preprocess image for model input
def preprocess_image(image):
    target_shape = input_shape  # Match the model's expected input shape
    image = cv2.resize(image, (target_shape[1], target_shape[0]))  # Resize to (width, height)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Detect face in the image using Haar Cascade
def detect_face(image):
    # Load pre-trained Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Adjust parameters for better face detection
    faces = face_cascade.detectMultiScale(
         gray,
        scaleFactor=1.05,  # Slightly reduce the scale factor for more sensitive detection
        minNeighbors=3,    # Reduce this to allow more face detections
        minSize=(30, 30)
    )
    if len(faces) == 0:
        return None

    # Crop the first detected face
    x, y, w, h = faces[0]
    cropped_face = image[y:y + h, x:x + w]
    return cropped_face

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')
    image_np = np.array(image)

    # Detect and crop the face
    face = detect_face(image_np)
    if face is None:
        return jsonify({'error': 'No face detected in the image'}), 400

    # Preprocess the cropped face
    processed_face = preprocess_image(face)

    # Predict with the model
    predictions = model.predict(processed_face)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    # Convert uploaded image to base64 for displaying
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({
        'skin_tone': skin_tone_labels.get(predicted_class, "Unknown"),
        'confidence': float(confidence),
        'uploaded_image': image_base64
    })

if __name__ == '__main__':
    app.run(debug=True)
