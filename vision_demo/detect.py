import cv2
from ultralytics import YOLO
import os

# --- PATH CONFIGURATION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# Loading your custom 'best.pt' if you have it, otherwise fallback to yolov8n
model_path = os.path.join(current_dir, "best.pt") if os.path.exists("best.pt") else "yolov8n.pt"
video_path = os.path.join(current_dir, "traffic_video.mp4")

model = YOLO(model_path)

# --- LINE CROSSING CONFIG ---
# Define a horizontal line in the middle of the frame
line_y = 400 
counter = 0
already_counted = []

cap = cv2.VideoCapture(video_path)

print(f"Using Model: {model_path}")
print("Starting Line-Crossing Detection...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run tracking (persist=True keeps track of the same car across frames)
    results = model.track(frame, persist=True, device='cpu')

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        
        for box, id in zip(boxes, ids):
            # Get center point of the bottom of the bounding box
            cx = int((box[0] + box[2]) / 2)
            cy = int(box[3]) 

            # Draw center point
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

            # Check if vehicle crossed the line
            if line_y - 5 < cy < line_y + 5:
                if id not in already_counted:
                    counter += 1
                    already_counted.append(id)

    # UI Visuals
    # Draw the "Trigger Line"
    cv2.line(frame, (0, line_y), (1280, line_y), (0, 0, 255), 3)
    cv2.putText(frame, f"TOTAL VEHICLES PASSED: {counter}", (50, 50), 
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Traffic Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
