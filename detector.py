# detector.py
from ultralytics import YOLO
import cv2

class Detector:
    def __init__(self, model_path="yolov8n.pt", conf=0.5):
        """
        Initialize the YOLO model for person detection.
        model_path: path to YOLO model
        conf: minimum confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        """
        Takes a frame from the camera and returns:
        - list of bounding boxes (x1, y1, x2, y2)
        - list of confidence scoresc
        """
        # Run YOLO inference (class 0 = person)
        results = self.model(frame, conf=self.conf, classes=[0])

        boxes = []
        confidences = []

        # Extract bounding boxes and confidence scores
        for r in results[0].boxes:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            conf = float(r.conf[0])

            boxes.append([x1, y1, x2, y2])
            confidences.append(conf)

        return boxes, confidences


# Standalone test using laptop webcam
if __name__ == "__main__":
    detector = Detector()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection on the frame
        boxes, confs = detector.detect(frame)

        # Draw boxes for testing purposes
        for (x1, y1, x2, y2), conf in zip(boxes, confs):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Detector Test", frame)

        # Press 'q' to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()