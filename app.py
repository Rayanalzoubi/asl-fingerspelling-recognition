from flask import Flask, render_template, request
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from collections import Counter
from mediapipe_handler import HandExtractor

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the RGB model
MODEL_PATH = "asl_efficientnet_best_finetuned.keras"
model = load_model(MODEL_PATH, compile=False)

# Class names (adjust to match your training if needed)
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]

# Mediapipe object for extracting the hand
extractor = HandExtractor()

# -----------------------------
# 🔹 Prepare the image for input
# -----------------------------
def preprocess_image(img):
    if img is None:
        raise ValueError("❌ الصورة غير صالحة.")

    # Convert to grayscale because the model expects a single channel only
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to match the model (225x225)
    img = cv2.resize(img, (225, 225))

    # Add the single channel dimension (1)
    img = np.expand_dims(img, axis=-1)

    # Add the fourth dimension (batch dimension)
    img = np.expand_dims(img, axis=0)

    # Normalize to [0,1]
    img = img / 255.0
    return img

# -----------------------------
# 🔹 Home page
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html')

# -----------------------------
# 🔹 Upload and analyze a file (image/video)
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    # Sanitize the name so uploads cannot write outside UPLOAD_FOLDER
    filename = secure_filename(file.filename) if file else ""
    if not filename:
        return "⚠️ لم يتم رفع أي ملف."

    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)
    ext = os.path.splitext(filename)[1].lower()

    if ext in ['.jpg', '.jpeg', '.png']:
        return analyze_image(path)
    elif ext in ['.mp4', '.avi', '.mov']:
        return analyze_video(path)
    else:
        return "⚠️ نوع الملف غير مدعوم."

# -----------------------------
# 🔹 Analyze a single image
# -----------------------------
def analyze_image(path):
    img = cv2.imread(path)
    hand_crop = extractor.extract_hand_roi(img)
    if hand_crop is None:
        return render_template('index.html', prediction="⚠️ لم يتم العثور على يد في الصورة.", image_path=path)

    X = preprocess_image(hand_crop)
    preds = model.predict(X)
    label = LABELS[np.argmax(preds)]
    confidence = float(np.max(preds) * 100)
    return render_template('index.html', prediction=label, confidence=round(confidence, 2), image_path=path)

# -----------------------------
# 🔹 Analyze a full video
# -----------------------------
def analyze_video(path):
    cap = cv2.VideoCapture(path)
    frame_predictions = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 5 != 0:  # only every 5th frame
            continue

        hand_crop = extractor.extract_hand_roi(frame)
        if hand_crop is not None:
            X = preprocess_image(hand_crop)
            preds = model.predict(X)
            label = LABELS[np.argmax(preds)]
            frame_predictions.append(label)

    cap.release()
    if not frame_predictions:
        return render_template('index.html', prediction="⚠️ لم يتم العثور على يد.", image_path=path)

    final_label = Counter(frame_predictions).most_common(1)[0][0]
    return render_template('index.html', prediction=final_label, image_path=path)

# -----------------------------
# 🔹 Run the application
# -----------------------------
if __name__ == '__main__':
    # Debug mode is off unless FLASK_DEBUG=1 is set
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
