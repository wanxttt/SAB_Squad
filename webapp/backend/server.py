import os
import sqlite3
import datetime
import time
import cv2
import pickle
import numpy as np
from flask import Flask, jsonify, Response
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. LOAD YOUR TRAINED MODEL & LABELS
MODEL_PATH = os.path.join(BACKEND_DIR, "detector_vgg16.h5")
LABEL_PATH = os.path.join(BACKEND_DIR, "lb.pickle")
VIDEO_PATH = os.path.join(BASE_DIR, "vision_demo", "traffic_video.mp4")
DB_PATH = os.path.join(BASE_DIR, "traffic_logs.db")

print("🤖 Loading Custom VGG16 Model...")
try:
    model = load_model(MODEL_PATH)
    lb = pickle.loads(open(LABEL_PATH, "rb").read())
    print(f"✅ Model Loaded! Classes: {lb.classes_}")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")
    print("Did you copy 'detector_vgg16.h5' and 'lb.pickle' to the backend folder?")
    model = None

# --- DATABASE SETUP ---
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS emergency_logs 
                        (id INTEGER PRIMARY KEY, timestamp TEXT, type TEXT, confidence REAL)''')
init_db()

def generate_frames():
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        if model:
            # --- YOUR PREDICT.PY LOGIC HERE ---
            # 1. Preprocess Image
            image = cv2.resize(frame, (224, 224))
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

            # 2. Predict
            # Returns: [bounding_box, class_probabilities]
            (boxPreds, labelPreds) = model.predict(image, verbose=0)
            
            # 3. Parse Output
            (startX, startY, endX, endY) = boxPreds[0]
            
            # Find the class with max probability
            i = np.argmax(labelPreds, axis=1)[0]
            label = lb.classes_[i]
            confidence = labelPreds[0][i]

            # 4. Logic: Check if Emergency
            # (Assuming your pickle classes are something like ['emergency', 'non_emergency'])
            is_emergency = "emergency" in label.lower() and "non" not in label.lower()

            # 5. Draw on Frame
            if confidence > 0.5: # Only show if confident
                h, w = frame.shape[:2]
                
                # Scale coordinates back to original frame size
                startX = int(startX * w)
                startY = int(startY * h)
                endX = int(endX * w)
                endY = int(endY * h)
                
                color = (0, 0, 255) if is_emergency else (0, 255, 0)
                label_text = f"{label.upper()}: {confidence*100:.2f}%"

                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.putText(frame, label_text, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 6. Log to Database if Emergency
                if is_emergency:
                    with sqlite3.connect(DB_PATH) as conn:
                        conn.execute("INSERT INTO emergency_logs (timestamp, type, confidence) VALUES (?, ?, ?)",
                                  (datetime.datetime.now(), label, float(confidence)))
                        conn.commit()

        # Encode for Web
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/logs')
def get_logs():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM emergency_logs ORDER BY id DESC LIMIT 5")
        return jsonify(cursor.fetchall())

@app.route('/api/status')
def get_status():
    # Helper to read the SUMO json file
    try:
        data_path = os.path.join(BASE_DIR, "live_data", "traffic_stats.json")
        with open(data_path, 'r') as f:
            data = json.load(f)
    except:
        data = {"waiting_cars": {"north": 0, "south": 0, "east": 0, "west": 0}}
        
    return jsonify(data)

if __name__ == '__main__':
    print(f"🔥 Custom VGG16 Server Running on http://localhost:5000")
    app.run(debug=True, port=5000, threaded=True)