import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import json
import os
import datetime

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Define comprehensive landmark sets
UPPER_FACE_POINTS = [10, 67, 109, 338, 297, 332, 251, 301]  # Forehead and temples
LOWER_FACE_POINTS = [164, 167, 165, 92, 186, 57, 43, 106]  # Mouth and chin
NOSE_BRIDGE_POINTS = [6, 197, 195, 5, 4]  # Nose bridge and tip
PERIMETER_POINTS = [234, 454, 365, 397, 288, 361, 323, 103]  # Face outline
EYE_POINTS = [33, 133, 159, 145, 386, 374]  # Eyes

# Combine all critical points
CRITICAL_POINTS = list(set(
    UPPER_FACE_POINTS + 
    LOWER_FACE_POINTS + 
    NOSE_BRIDGE_POINTS + 
    PERIMETER_POINTS +
    EYE_POINTS
))

print(f"Using {len(CRITICAL_POINTS)} critical landmarks for detection")

# Texture analysis parameters
PATCH_SIZE = 11
TEXTURE_THRESHOLD = 22  # Lower = more sensitive to coverage
DEPTH_THRESHOLD = 0.07  # Higher = more sensitive to occlusion

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Confidence smoothing
confidence_queue = deque(maxlen=15)

# Debug visualization settings
DEBUG_MODE = True  # Set to False to disable debug visuals

# Create directory for captured frames
os.makedirs("captured_frames", exist_ok=True)

# Alert tracking variables
alert_data = []
current_alert = None
frames_to_capture = 5
captured_frames = 0

# JSON file path
json_file = "face_cover_alerts.json"

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue
        
    # Mirror the frame for more intuitive interaction
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Process frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    face_status = "NO FACE DETECTED"
    color = (255, 0, 0)  # Blue for no face
    coverage_confidence = 0.0
    avg_confidence = 0.0
    
    if results.multi_face_landmarks:
        for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
            # Analyze landmarks
            visible_points = 0
            depth_variance = []
            texture_failures = 0
            
            # Draw face mesh for visualization (optional)
            if DEBUG_MODE:
                mp.solutions.drawing_utils.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_tesselation_style()
                )
            
            # Check all critical points
            for idx in CRITICAL_POINTS:
                if idx >= len(face_landmarks.landmark):
                    continue
                    
                landmark = face_landmarks.landmark[idx]
                x, y = int(landmark.x * w), int(landmark.y * h)
                
                # Always record depth (z-coordinate)
                depth_variance.append(landmark.z)
                
                # Skip points outside frame
                if not (0 <= x < w and 0 <= y < h):
                    if DEBUG_MODE:
                        cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)  # Yellow for out-of-frame
                    continue
                
                # Perform texture analysis around landmark
                y1, y2 = max(0, y-PATCH_SIZE//2), min(h, y+PATCH_SIZE//2+1)
                x1, x2 = max(0, x-PATCH_SIZE//2), min(w, x+PATCH_SIZE//2+1)
                
                if y2 > y1 and x2 > x1:
                    patch = frame[y1:y2, x1:x2]
                    
                    if patch.size > 0:
                        # Convert to grayscale and calculate texture variation
                        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                        texture_std = np.std(gray)
                        
                        # Determine if this point shows signs of coverage
                        is_covered = texture_std < TEXTURE_THRESHOLD
                        
                        if DEBUG_MODE:
                            # Draw debug visuals
                            if idx in LOWER_FACE_POINTS:
                                # Red for covered areas, green for clear
                                point_color = (0, 0, 255) if is_covered else (0, 255, 0)
                                cv2.circle(frame, (x, y), 4, point_color, -1)
                            
                            # Show texture value
                            cv2.putText(frame, f"{texture_std:.1f}", 
                                       (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.3, (255, 255, 255), 1)
                        
                        if is_covered:
                            texture_failures += 1
                        else:
                            visible_points += 1
                
                # Draw all critical points
                if DEBUG_MODE:
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)  # Yellow for critical points
            
            # Calculate coverage metrics
            total_points = len(CRITICAL_POINTS)
            
            # Depth variance analysis
            if depth_variance:
                depth_range = max(depth_variance) - min(depth_variance)
                depth_score = min(1.0, depth_range / DEPTH_THRESHOLD)
            else:
                depth_score = 0
                
            # Visibility ratio (points that passed texture check)
            visibility_ratio = visible_points / total_points if total_points > 0 else 0
            
            # Texture failure ratio
            texture_failure_ratio = texture_failures / total_points if total_points > 0 else 0
            
            # Combined coverage confidence
            coverage_confidence = (1 - visibility_ratio) * 0.5 + texture_failure_ratio * 0.3 + depth_score * 0.2
            confidence_queue.append(coverage_confidence)
            avg_confidence = sum(confidence_queue) / len(confidence_queue) if confidence_queue else 0
            
            # Determine face status
            if avg_confidence >= 0.5:
                face_status = "⚠️ FACE COVERED"
                color = (0, 0, 255)  # Red
                
                # Start new alert if not already capturing
                if current_alert is None:
                    current_alert = {
                        "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "frames": [],
                        "max_confidence": avg_confidence
                    }
                    captured_frames = 0
                
                # Capture frames if we're in an active alert
                if current_alert and captured_frames < frames_to_capture:
                    # Generate unique filename
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"captured_frames/alert_{timestamp}.jpg"
                    
                    # Save frame
                    cv2.imwrite(filename, frame)
                    
                    # Add to alert data
                    current_alert["frames"].append({
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                        "confidence": avg_confidence,
                        "filename": filename
                    })
                    
                    # Update max confidence
                    current_alert["max_confidence"] = max(current_alert["max_confidence"], avg_confidence)
                    
                    captured_frames += 1
                    print(f"Captured frame {captured_frames}/{frames_to_capture}")
                    
                    # If we've captured all frames, save the alert
                    if captured_frames >= frames_to_capture:
                        alert_data.append(current_alert)
                        with open(json_file, "w") as f:
                            json.dump(alert_data, f, indent=4)
                        print(f"Alert saved with {len(current_alert['frames'])} frames")
                        current_alert = None
            
            else:
                face_status = "✅ FACE CLEAR"
                color = (0, 255, 0)  # Green
                
                # Reset alert tracking if face is clear
                if current_alert:
                    # Save partial alert if we have any frames
                    if current_alert["frames"]:
                        alert_data.append(current_alert)
                        with open(json_file, "w") as f:
                            json.dump(alert_data, f, indent=4)
                        print(f"Partial alert saved with {len(current_alert['frames'])} frames")
                    current_alert = None
                
            # Show detailed metrics
            if DEBUG_MODE:
                cv2.putText(frame, f"Visible: {visible_points}/{total_points}", (20, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)
                cv2.putText(frame, f"Texture Failures: {texture_failures}", (20, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)
                cv2.putText(frame, f"Depth Score: {depth_score:.2f}", (20, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)
    
    # No face detected
    else:
        confidence_queue.clear()
        
        # Reset alert tracking if no face
        if current_alert:
            # Save partial alert if we have any frames
            if current_alert["frames"]:
                alert_data.append(current_alert)
                with open(json_file, "w") as f:
                    json.dump(alert_data, f, indent=4)
                print(f"Partial alert saved with {len(current_alert['frames'])} frames")
            current_alert = None
    
    # Show status
    cv2.putText(frame, face_status, (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    
    # Confidence meter
    meter_width = 200
    confidence_width = int(avg_confidence * meter_width)
    cv2.rectangle(frame, (20, 60), (20 + meter_width, 80), (50, 50, 50), -1)
    cv2.rectangle(frame, (20, 60), (20 + confidence_width, 80), color, -1)
    cv2.putText(frame, f"Confidence: {avg_confidence:.2f}", 
               (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Display frame
    cv2.imshow("Face Cover Detection", frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        # Save any active alert before exiting
        if current_alert and current_alert["frames"]:
            alert_data.append(current_alert)
            with open(json_file, "w") as f:
                json.dump(alert_data, f, indent=4)
            print(f"Final alert saved with {len(current_alert['frames'])} frames")
        break
    elif key == ord('d'):
        DEBUG_MODE = not DEBUG_MODE

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Print summary
print(f"\nDetection completed. {len(alert_data)} alerts recorded.")
print(f"Alert data saved to: {json_file}")
print(f"Captured frames saved to: captured_frames/")