import cv2
from ultralytics import YOLO
import os

# --- CONFIGURATION ---
# Get the absolute path to the video file next to this script
current_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(current_dir, "traffic_video.mp4")

# --- LOAD THE AI MODEL ---
print("Loading YOLOv8 AI Model... (this might take a moment first time)")
# We use 'yolov8n.pt' (Nano). It's the fastest, smallest model.
# It will download automatically the first time you run this.
model = YOLO('yolov8n.pt') 

# --- OPEN VIDEO SOURCE ---
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ Error: Could not open video file: {video_path}")
    exit()

print("Video opened successfully. Starting detection loop...")
print("Press 'Q' on your keyboard to stop.")

# --- MAIN PROCESS LOOP ---
while True:
    # 1. Read a frame from the video
    success, frame = cap.read()
    if not success:
        print("End of video reached.")
        break # Stop loop if video ends

    # 2. Run AI inference on the frame
    # stream=True makes it faster for video generators
    results = model(frame, stream=True)

    # 3. Process the results
    for r in results:
        # The model automatically draws the boxes and labels onto the frame for us
        annotated_frame = r.plot()

        # --- Calculate Counts (Optional Data gathering) ---
        # Get the detected classes indices
        classes = r.boxes.cls.cpu().numpy()
        # Count specific vehicle types (based on COCO dataset indices: 2=car, 3=motorcycle, 5=bus, 7=truck)
        car_count = (classes == 2).sum()
        truck_count = (classes == 7).sum()
        bus_count = (classes == 5).sum()
        
        # Add a counter overlay to the video
        info_text = f"Cars: {car_count} | Trucks: {truck_count} | Buses: {bus_count}"
        cv2.putText(annotated_frame, info_text, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


    # 4. Display the resulting frame
    cv2.imshow("Traffic-Pulse Vision Demo (YOLOv8)", annotated_frame)

    # 5. Wait for 'Q' key to exit Wait 1ms between frames
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()
print("Vision demo finished.")