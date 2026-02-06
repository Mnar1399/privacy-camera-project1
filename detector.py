# detector.py
import cv2
import numpy as np
from ultralytics import YOLO
import time


class PersonDetector:
    """
    Enhanced person detector with temporal filtering and better validation.
    """
    
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.4, 
                 iou_threshold=0.45, min_person_area=1500):
        """
        Initialize the detector with better filtering.
        
        Args:
            model_path: Path to YOLO model
            conf_threshold: Confidence threshold
            iou_threshold: NMS threshold
            min_person_area: Minimum area for person detection
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.min_person_area = min_person_area
        self.person_class_id = 0
        
        # Temporal filtering
        self.last_detections = []
        self.filter_weight = 0.7  # How much to trust new vs old detections
        
    def detect(self, frame):
        """
        Detect persons with enhanced filtering.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (boxes, confidences)
        """
        # Run inference
        results = self.model(
            frame, 
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[self.person_class_id],
            device="cpu",
            verbose=False
        )[0]
        
        boxes = []
        confidences = []
        
        if results.boxes is not None:
            for box in results.boxes:
                # Double-check it's a person
                if int(box.cls[0]) != 0:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # Area filter
                area = (x2 - x1) * (y2 - y1)
                if area < self.min_person_area:
                    continue
                
                # Aspect ratio filter (persons are typically taller than wide)
                width = x2 - x1
                height = y2 - y1
                aspect_ratio = width / height if height > 0 else 0
                
                if aspect_ratio > 2.0 or aspect_ratio < 0.3:
                    continue
                
                boxes.append([x1, y1, x2, y2])
                confidences.append(conf)
        
        # Apply temporal filtering for smoother results
        if len(self.last_detections) > 0:
            boxes, confidences = self._temporal_filter(boxes, confidences)
        
        self.last_detections = boxes.copy()
        return boxes, confidences
    
    def _temporal_filter(self, current_boxes, current_confs):
        """Apply temporal smoothing to detections."""
        if not current_boxes:
            return [], []
        
        filtered_boxes = []
        filtered_confs = []
        
        for i, (box, conf) in enumerate(zip(current_boxes, current_confs)):
            matched = False
            
            for prev_box in self.last_detections:
                iou = self._iou(box, prev_box)
                if iou > 0.3:  # Match found
                    # Smooth the position
                    blended_box = [
                        int(self.filter_weight * box[0] + (1 - self.filter_weight) * prev_box[0]),
                        int(self.filter_weight * box[1] + (1 - self.filter_weight) * prev_box[1]),
                        int(self.filter_weight * box[2] + (1 - self.filter_weight) * prev_box[2]),
                        int(self.filter_weight * box[3] + (1 - self.filter_weight) * prev_box[3]),
                    ]
                    filtered_boxes.append(blended_box)
                    filtered_confs.append(conf)
                    matched = True
                    break
            
            if not matched and conf > 0.5:  # New high-confidence detection
                filtered_boxes.append(box)
                filtered_confs.append(conf)
        
        return filtered_boxes, filtered_confs
    
    def _iou(self, box1, box2):
        """Calculate IoU."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0