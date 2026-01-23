import os
import sqlite3
import datetime
import time
import json
import cv2
import threading
import numpy as np
import queue
import random
import sounddevice as sd
import librosa
from flask import Flask, jsonify, Response
from flask_cors import CORS
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# (Removed onnxruntime import since we are using .pt)

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "traffic_logs.db")

# Paths
# --- CHANGE: Points to the .pt FILE in your backend folder ---
YOLO_PATH = os.path.join(BACKEND_DIR, "best.pt") 
AUDIO_MODEL_PATH = os.path.join(BACKEND_DIR, "CNN_Model.keras") 
VIDEO_PATH = os.path.join(BACKEND_DIR, "traffic_video.mp4")

# --- TUNING ---
SILENCE_THRESHOLD = 0.1
CONFIDENCE_THRESHOLD = 0.5
VISUAL_CONFIDENCE = 0.6
Q_LEARNING_EPISODES = 500

# --- GLOBAL SYSTEM STATE ---
SYSTEM_STATUS = {"visual_emergency": False, "audio_emergency": False}
audio_queue = queue.Queue()

# ==========================================
# 1. TRAFFIC NETWORK LOGIC
# ==========================================
class Intersection:
    def __init__(self, id, is_real=False):
        self.id = id
        self.is_real = is_real
        self.queues = {"north": 5, "south": 5, "east": 5, "west": 5} 
        self.green_phase = "north"
    
    def step(self):
        if not self.is_real:
            if random.random() < 0.3: 
                self.queues[random.choice(["north", "south", "east", "west"])] += 1
        if self.queues[self.green_phase] > 0:
            self.queues[self.green_phase] -= 0.5 
            if self.queues[self.green_phase] < 0: self.queues[self.green_phase] = 0

class TrafficNetwork:
    def __init__(self):
        self.nodes = [Intersection(0, True), Intersection(1), Intersection(2), Intersection(3)]
        self.emergency_mode = False
        self.episode_count = 0
        self.avg_wait_history = []

    def optimize_step(self):
        self.episode_count += 1
        total_wait = 0
        for node in self.nodes:
            wait = sum(node.queues.values())
            total_wait += wait
            if self.emergency_mode:
                node.green_phase = "north"
                continue
            if node.queues[node.green_phase] < 1 or random.random() < 0.05:
                node.green_phase = max(node.queues, key=node.queues.get)
            node.step()
        if self.episode_count % 10 == 0:
            self.avg_wait_history.append(total_wait / 4)
            if len(self.avg_wait_history) > 20: self.avg_wait_history.pop(0)

network = TrafficNetwork()

def simulation_loop():
    while True:
        network.optimize_step()
        time.sleep(1) 

sim_thread = threading.Thread(target=simulation_loop, daemon=True)
sim_thread.start()

# ==========================================
# 2. AI MODELS (Standard .pt Loader)
# ==========================================
try:
    # Load the PyTorch model
    vision_model = YOLO(YOLO_PATH) 
    print(f"✅ Custom Model Loaded: {YOLO_PATH}")
    
    # Print classes to confirm what your model can detect
    print(f"📋 Model Classes: {vision_model.names}")
    
except Exception as e: 
    print(f"⚠️ Failed to load Custom Model: {e}")
    print("⬇️ Downloading standard YOLOv8n as fallback...")
    vision_model = YOLO("yolov8n.pt")

audio_model = None
try:
    if os.path.exists(AUDIO_MODEL_PATH) and not os.path.isdir(AUDIO_MODEL_PATH):
        audio_model = load_model(AUDIO_MODEL_PATH)
        print("✅ Audio Loaded")
except Exception as e: print(f"❌ Audio Error: {e}")

labelencoder = LabelEncoder()
labelencoder.classes_ = np.array(['ambulance', 'firetruck', 'traffic']) 

# ==========================================
# 3. AUDIO PROCESSING
# ==========================================
def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

def audio_processor():
    if not audio_model: return
    SAMPLE_RATE = 22050
    DURATION = 3
    BLOCK_SIZE = int(SAMPLE_RATE * DURATION)
    
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE):
        while True:
            try:
                audio = audio_queue.get().flatten()
                vol = np.sqrt(np.mean(audio**2)) * 10
                if vol < SILENCE_THRESHOLD:
                    SYSTEM_STATUS["audio_emergency"] = False
                    if not SYSTEM_STATUS["visual_emergency"]: network.emergency_mode = False
                    continue 

                mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=80).T, axis=0)
                pred = audio_model.predict(mfcc.reshape(1, 80, 1), verbose=0)[0]
                idx = np.argmax(pred)
                label = labelencoder.inverse_transform([idx])[0]
                conf = pred[idx]
                
                print(f"🔊 Vol: {vol:.1f} | Heard: {label.upper()} ({conf:.2f})")

                if label in ['ambulance', 'firetruck'] and conf > CONFIDENCE_THRESHOLD:
                    print(f"🚨 SIREN DETECTED: {label.upper()}")
                    SYSTEM_STATUS["audio_emergency"] = True
                    network.emergency_mode = True
                else:
                    SYSTEM_STATUS["audio_emergency"] = False
                    if not SYSTEM_STATUS["visual_emergency"]: network.emergency_mode = False

            except Exception as e: print(f"Audio Error: {e}")

threading.Thread(target=audio_processor, daemon=True).start()

# ==========================================
# 4. VISION PROCESSING
# ==========================================
def generate_frames():
    print(f"🎥 ATTEMPTING TO LOAD VIDEO FROM: {VIDEO_PATH}")
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("❌ ERROR: Could not open video file. Check the path!")
        while True:
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank_frame, "VIDEO FILE NOT FOUND", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(blank_frame, "Check Server Console", (100, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            _, buffer = cv2.imencode('.jpg', blank_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1)

    while True:
        success, frame = cap.read()
        if not success: 
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        real_counts = {"north": 0, "south": 0, "east": 0, "west": 0}
        visual_trigger = False
        
        if vision_model:
            results = vision_model(frame, verbose=False)[0]
            height, width, _ = frame.shape
            cx, cy = width//2, height//2
            
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = vision_model.names[int(box.cls[0])]
                bcx, bcy = (x1+x2)//2, (y1+y2)//2
                
                if bcy < cy: direction = "north" if bcx < cx else "east"
                else: direction = "west" if bcx < cx else "south"
                
                # Count traffic
                if label in ['car', 'truck', 'bus', 'motorcycle', 'ambulance']: 
                    real_counts[direction] += 1
                
                # --- Emergency Detection ---
                # Check for your custom 'ambulance' class OR standard truck/bus as proxy
                is_custom_ambulance = (label == 'ambulance' and float(box.conf[0]) > VISUAL_CONFIDENCE)
                is_proxy_emergency = (label in ['truck', 'bus'] and float(box.conf[0]) > VISUAL_CONFIDENCE)
                
                if is_custom_ambulance or is_proxy_emergency:
                    visual_trigger = True
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 3)
                    
                    if is_custom_ambulance:
                        label_text = f"CUSTOM: {label.upper()}"
                    else:
                        label_text = f"VISUAL: {label.upper()}"
                        
                    cv2.putText(frame, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                else:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)

        network.nodes[0].queues = real_counts
        SYSTEM_STATUS["visual_emergency"] = visual_trigger
        
        if visual_trigger or SYSTEM_STATUS["audio_emergency"]: 
            network.emergency_mode = True

        if network.emergency_mode:
            status_text = "GREEN WAVE ACTIVE"
            color = (0, 0, 255)
        else:
            status_text = "SYSTEM NORMAL"
            color = (0, 255, 0)
            
        cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ==========================================
# 5. API ENDPOINTS
# ==========================================
@app.route('/api/status')
def get_status():
    node0 = network.nodes[0]
    return jsonify({
        "waiting_cars": node0.queues,
        "current_green_phase": node0.green_phase,
        "avg_wait_time": 15, 
        "emergency_detected": network.emergency_mode,
        "audio_alert": SYSTEM_STATUS["audio_emergency"]
    })

@app.route('/api/network_status')
def get_network():
    nodes_data = []
    for node in network.nodes:
        nodes_data.append({
            "id": node.id,
            "queues": node.queues,
            "green_phase": node.green_phase,
            "is_real": node.is_real
        })
    return jsonify({
        "nodes": nodes_data,
        "emergency": network.emergency_mode,
        "audio_alert": SYSTEM_STATUS["audio_emergency"],
        "learning_metrics": network.avg_wait_history
    })

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/logs')
def get_logs():
    return jsonify([]) 

if __name__ == '__main__':
    print("🔥 TRAFFIC NETWORK SIMULATION STARTED")
    app.run(debug=True, port=5000, threaded=True, use_reloader=False)
