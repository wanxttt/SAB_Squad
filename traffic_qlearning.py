import numpy as np
import random
import time
import os
import torch
from torch import nn
import librosa
from ultralytics import YOLO

# ─── GLOBAL DEVICE SETTING (Fixes scope error) ────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Constants ─────────────────────────────────────────────
NUM_ROADS = 4  # 0: North, 1: East, 2: South, 3: West
PAIRS = [[0, 2], [1, 3]]  # Opposite pairs for simultaneous green
DURATIONS = [30, 60, 90]  # Short, medium, long (seconds)
YELLOW_TIME = 5
EMERGENCY_GREEN_MIN = 30
DETECTION_THRESHOLD = 0.6
NUM_DENSITY_LEVELS = 3  # Low=0, Med=1, High=2
NUM_STATES = NUM_DENSITY_LEVELS ** NUM_ROADS
NUM_ACTIONS = len(PAIRS) * len(DURATIONS)  # 2 pairs * 3 durations = 6

# ─── Q-Learning Params ─────────────────────────────────────
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
EPISODES = 500
MAX_STEPS = 100

# ─── State Encoding ────────────────────────────────────────
def state_to_index(densities):
    index = 0
    for d in densities:
        index = index * NUM_DENSITY_LEVELS + d
    return index

def index_to_state(index):
    densities = []
    for _ in range(NUM_ROADS):
        densities.append(index % NUM_DENSITY_LEVELS)
        index //= NUM_DENSITY_LEVELS
    densities.reverse()
    return densities

# ─── Simulate Traffic Step ─────────────────────────────────
def simulate_traffic_step(densities, pair_index, duration_index):
    duration = DURATIONS[duration_index]
    green_roads = PAIRS[pair_index]
    new_densities = densities.copy()
    
    for road in range(NUM_ROADS):
        # Arrival: random increase
        arrival = random.choice([0, 1, 2]) if random.random() < 0.4 else 0
        new_densities[road] = min(2, new_densities[road] + arrival)
        
        # Departure: only on green
        if road in green_roads:
            departure = min(new_densities[road], random.randint(1, 2))  # Release 1-2 units
            new_densities[road] -= departure
    
    return new_densities

# ─── Reward: -avg wait (density on red * time) ─────────────
def calculate_reward(densities, pair_index, duration_index):
    duration = DURATIONS[duration_index]
    red_roads = [r for r in range(NUM_ROADS) if r not in PAIRS[pair_index]]
    total_wait = sum(densities[r] for r in red_roads) * duration
    avg_wait = total_wait / NUM_ROADS
    return -avg_wait

# ─── Load Vision Model (YOLO ambulance) ────────────────────
def load_vision_model():
    # Make sure 'best.pt' is in the same folder or provide full path
    model_path = "best.pt" 
    if not os.path.exists(model_path):
        print(f"⚠️ Warning: {model_path} not found. Using yolov8n.pt")
        return YOLO("yolov8n.pt")
    return YOLO(model_path)

# ─── Load Audio Model (SirenCNN siren) ─────────────────────
class SirenCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(40, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
        )
        # Dummy pass to calculate flat size
        dummy = torch.zeros(1, 40, 173)
        out = self.conv(dummy)
        flat = out.view(1, -1).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(flat, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        # x shape: [batch, 40, time] -> conv expects [batch, channels, length]
        # If input is already [batch, 40, time], no permute needed for 1D Conv
        # Adjust based on your specific training data shape
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def load_audio_model():
    model = SirenCNN().to(device)
    path = "best_siren_audio.pt"
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            print("✅ Audio Model Loaded")
        except Exception as e:
            print(f"❌ Audio Load Error: {e}")
    else:
        print(f"⚠️ Warning: {path} not found.")
    return model

# ─── Emergency Detection (Vision + Audio) ──────────────────
def detect_emergency(road_id, vision_model, audio_model, frame, audio_clip):
    # Vision: ambulance detected
    # Note: 'frame' here is a dummy placeholder in the simulation loop below. 
    # In real usage, you'd pass a real cv2 frame.
    has_ambulance = False
    if isinstance(frame, np.ndarray) and frame.sum() > 0: # Simple check if frame is real
        vision_results = vision_model(frame, verbose=False, conf=DETECTION_THRESHOLD)
        # Check if 'ambulance' class exists in your model's names
        # Assuming class ID for ambulance is known or checking label names
        for box in vision_results[0].boxes:
            cls_id = int(box.cls[0])
            if vision_model.names[cls_id] == 'ambulance':
                has_ambulance = True
                break
    
    # Audio: siren detected
    has_siren = False
    if len(audio_clip) > 0:
        try:
            # Process audio to MFCC
            mfcc = librosa.feature.mfcc(y=audio_clip, sr=22050, n_mfcc=40)
            # Pad/Crop to 173 width to match model input
            if mfcc.shape[1] < 173:
                pad_width = 173 - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)))
            else:
                mfcc = mfcc[:, :173]
                
            mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                out = audio_model(mfcc_tensor)
                # Assuming index 1 is Siren
                siren_prob = torch.softmax(out, dim=1)[0][1].item()
            has_siren = siren_prob > DETECTION_THRESHOLD
        except Exception as e:
            # print(f"Audio processing error: {e}")
            pass
    
    return has_ambulance or has_siren # Logic: Either visual OR audio triggers

# ─── Q-Learning Training (Run Once to Learn Q-Table) ───────
def train_q_learning():
    global EPSILON  # <--- MOVED HERE TO FIX YOUR ERROR
    
    print("🧠 Training Q-Learning Model...")
    Q = np.zeros((NUM_STATES, NUM_ACTIONS))
    episode_waits = []
    
    for episode in range(EPISODES):
        densities = [random.randint(0, 2) for _ in range(NUM_ROADS)]
        state = state_to_index(densities)
        total_wait = 0
        
        for step in range(MAX_STEPS):
            if random.random() < EPSILON:
                action = random.randint(0, NUM_ACTIONS - 1)
            else:
                action = np.argmax(Q[state])
            
            pair_index = action // len(DURATIONS)
            duration_index = action % len(DURATIONS)
            
            new_densities = simulate_traffic_step(densities, pair_index, duration_index)
            new_state = state_to_index(new_densities)
            
            reward = calculate_reward(densities, pair_index, duration_index)
            total_wait -= reward  # Since reward = -wait
            
            Q[state, action] += ALPHA * (reward + GAMMA * np.max(Q[new_state]) - Q[state, action])
            
            state = new_state
            densities = new_densities
        
        episode_waits.append(total_wait / MAX_STEPS)
        
        EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
    
    np.save("q_table.npy", Q)
    print("✅ Q-table saved to 'q_table.npy'")
    return episode_waits

# ─── Main Simulation Loop (With Emergency & Q-Learning) ────
def run_traffic_simulation():
    # Load models
    vision_model = load_vision_model()
    audio_model = load_audio_model()
    
    # Load pre-trained Q-table
    if os.path.exists("q_table.npy"):
        Q = np.load("q_table.npy")
        print("✅ Loaded existing Q-table")
    else:
        train_q_learning()
        Q = np.load("q_table.npy")
    
    # State vars
    densities = [0] * NUM_ROADS  # Start low
    lights = ["RED"] * NUM_ROADS
    current_pair = 0
    timer = 0
    emergency_queue = []  # (road, detect_time)
    emergency_active = False
    
    print("\n🚦 Simulation running... (Press Ctrl+C to stop)")
    
    try:
        while True:
            # Simulate detections (In a real app, these come from sensors)
            for road in range(NUM_ROADS):
                # Placeholder frame/audio - in real integration these would be live inputs
                frame = np.zeros((640, 640, 3), dtype=np.uint8) 
                audio_clip = np.random.rand(22050 * 4).astype(np.float32) 
                
                # Randomly trigger "emergency" for testing (5% chance)
                if random.random() < 0.05:  
                    if detect_emergency(road, vision_model, audio_model, frame, audio_clip):
                        if road not in [q[0] for q in emergency_queue]:
                            print(f"🚨 EMERGENCY DETECTED ON ROAD {road}")
                            emergency_queue.append((road, time.time()))
                            emergency_active = True
            
            if emergency_active:
                if emergency_queue:
                    priority_road = emergency_queue[0][0]
                    # Determine green pair based on emergency road
                    current_pair = 0 if priority_road in PAIRS[0] else 1
                    
                    # Force Green for priority pair
                    lights = ["GREEN" if r in PAIRS[current_pair] else "RED" for r in range(NUM_ROADS)]
                    
                    duration = EMERGENCY_GREEN_MIN
                    timer += 1
                    
                    # Simulation: Clear emergency after some time (randomly for test)
                    if timer > 5: # Assume cleared after 5 ticks in simulation
                        emergency_queue.pop(0)
                        if not emergency_queue:
                            emergency_active = False
                            timer = 0
                            print("✅ Emergency Cleared. Resuming Normal Logic.")
            else:
                # Normal Q-Learning Logic
                state = state_to_index(densities)
                action = np.argmax(Q[state])
                current_pair = action // len(DURATIONS)
                duration_index = action % len(DURATIONS)
                duration = DURATIONS[duration_index]
                
                if timer >= duration + YELLOW_TIME:
                    timer = 0
                elif timer >= duration:
                    lights = ["YELLOW" if r in PAIRS[current_pair] else "RED" for r in range(NUM_ROADS)]
                else:
                    lights = ["GREEN" if r in PAIRS[current_pair] else "RED" for r in range(NUM_ROADS)]
                
                densities = simulate_traffic_step(densities, current_pair, duration_index)
                timer += 1
            
            # Print Status
            print(f"Status: {lights} | Densities: {densities} | Emergency: {emergency_active}")
            
            time.sleep(1)  # 1 second tick
            
    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped.")

if __name__ == "__main__":
    run_traffic_simulation()