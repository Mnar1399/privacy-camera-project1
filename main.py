# main.py
import cv2
from detector import Detector
from tracker import Tracker
from privacy import blur_person

# ================== SOURCE ==================
CAMERA = 0   

# ================== INIT ==================
detector = Detector(conf=0.5)
tracker = Tracker()

cap = cv2.VideoCapture(CAMERA)

if not cap.isOpened():
    print("❌ Camera not opened")
    exit()

print("✅ Camera started")

# ================== MAIN LOOP ==================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame not received")
        break

    # ---------- DETECTION ----------
    boxes, confs = detector.detect(frame)

    detections = []
    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = box
        detections.append([x1, y1, x2, y2, conf])

    # ---------- TRACKING ----------
    tracked_people = tracker.update(detections, frame)

    # ---------- BLUR + ID ----------
    for track_id, x1, y1, x2, y2 in tracked_people:
        frame = blur_person(frame, x1, y1, x2, y2)

        cv2.putText(
            frame,
            f"ID: {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    cv2.imshow("Privacy Camera (Live)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ================== CLEANUP ==================
cap.release()
cv2.destroyAllWindows()

