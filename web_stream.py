import cv2
import time
from detector import PersonDetector
from tracker import Tracker
from privacy import blur_person

# التحكم من الويب
control = {
    "running": False,
    "privacy": True
}

def generate_frames():
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

    cap = None
    fps = 0
    frame_count = 0
    start_time = time.time()

    while True:
        # لو الكاميرا متوقفة
        if not control["running"]:
            if cap is not None:
                cap.release()
                cap = None
            time.sleep(0.1)
            continue

        # افتح الكاميرا فقط عند Start
        if cap is None:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                time.sleep(0.5)
                continue

        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        # Detection
        boxes, confidences = detector.detect(frame)

        display_frame = frame.copy()

        # Privacy Blur (نفس main.py)
        if control["privacy"]:
            for bbox in boxes:
                x1, y1, x2, y2 = map(int, bbox)
                display_frame = blur_person(display_frame, x1, y1, x2, y2)

        # Tracking
        tracks = tracker.update(frame, boxes, confidences)

        # رسم البوكس + ID
        for track_id, track_info in tracks.items():
            x1, y1, x2, y2 = map(int, track_info["bbox"])
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                display_frame,
                f"ID {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        # FPS
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed > 0:
            fps = frame_count / elapsed

        # Stats Box
        cv2.rectangle(display_frame, (5, 5), (260, 110), (0, 0, 0), -1)

        stats = [
            f"FPS: {fps:.1f}",
            f"Persons: {len(tracks)}",
            f"Privacy: {'ON' if control['privacy'] else 'OFF'}"
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

        # إرسال الفريم للويب
        _, buffer = cv2.imencode(".jpg", display_frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes +
            b"\r\n"
        )
