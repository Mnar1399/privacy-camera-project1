# main.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import time
from detector import PersonDetector
from tracker import Tracker
from privacy import blur_person


def main():
    detector = PersonDetector(
        model_path="yolov8n.pt",
        conf_threshold=0.4,
        min_person_area=1500
    )
    
    tracker = Tracker(
        max_disappeared=50,
        iou_threshold=0.3,
        use_appearance=True,
        appearance_weight=0.4
    )
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # FPS calculation
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    # Colors for different tracks
    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 0),    # Dark Blue
        (0, 128, 0),    # Dark Green
        (0, 0, 128),    # Dark Red
        (128, 128, 0),  # Olive
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
    ]
    
    # Privacy control
    privacy_enabled = True
    
    print("Starting person tracking...")
    print("Press 'q' to quit")
    print("Press 'p' to toggle privacy blur on/off")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Mirror the frame
        frame = cv2.flip(frame, 1)
        
        # Detect persons
        boxes, confidences = detector.detect(frame)
        
        # Create display frame
        display_frame = frame.copy()
        
        # Apply privacy blur if enabled
        if privacy_enabled:
            for bbox in boxes:
                x1, y1, x2, y2 = map(int, bbox)
                display_frame = blur_person(display_frame, x1, y1, x2, y2)
        
        # Update tracker
        tracks = tracker.update(frame, boxes, confidences)
        
        # Draw tracks on display frame
        for track_id, track_info in tracks.items():
            bbox = track_info['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Get color for this track
            color = colors[track_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID
            label = f"ID {track_id}"
            
            # Text background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            
            cv2.rectangle(
                display_frame, 
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Text
            cv2.putText(
                display_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Draw trajectory
            if 'trajectory' in track_info and len(track_info['trajectory']) > 1:
                trajectory = list(track_info['trajectory'])
                
                # Draw lines connecting points
                for i in range(1, len(trajectory)):
                    pt1 = (int(trajectory[i-1][0]), int(trajectory[i-1][1]))
                    pt2 = (int(trajectory[i][0]), int(trajectory[i][1]))
                    cv2.line(display_frame, pt1, pt2, color, 2)
                
                # Draw current position
                last_pt = (int(trajectory[-1][0]), int(trajectory[-1][1]))
                cv2.circle(display_frame, last_pt, 5, color, -1)
        
        # Calculate and display FPS
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed > 0:
            fps = frame_count / elapsed
        
        # Draw simple statistics
        cv2.rectangle(display_frame, (5, 5), (250, 90), (0, 0, 0), -1)
        
        stats = [
            f"FPS: {fps:.1f}",
            f"Persons: {len(tracks)}",
            f"Privacy: {'ON' if privacy_enabled else 'OFF'}"
        ]
        
        for i, stat in enumerate(stats):
            cv2.putText(
                display_frame,
                stat,
                (10, 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        
        # Show frame
        window_title = "Person Tracker"
        cv2.imshow(window_title, display_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('p'):
            privacy_enabled = not privacy_enabled
            status = "ON" if privacy_enabled else "OFF"
            print(f"Privacy blur: {status}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Tracking stopped.")


if __name__ == "__main__":
    main()