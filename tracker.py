# tracker.py
import cv2
import numpy as np
from collections import OrderedDict, deque
import time


class Tracker:
    """
    Enhanced tracker for consistent person IDs with re-identification.
    """
    
    def __init__(self, max_disappeared=50, iou_threshold=0.3, 
                 use_appearance=True, appearance_weight=0.4):
        """
        Initialize tracker with re-identification capabilities.
        
        Args:
            max_disappeared: Frames before track disappears (increased for re-id)
            iou_threshold: IoU threshold for matching
            use_appearance: Use appearance features for re-identification
            appearance_weight: Weight for appearance vs IoU in matching
        """
        self.next_id = 0
        self.tracks = OrderedDict()
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.use_appearance = use_appearance
        self.appearance_weight = appearance_weight
        
        # For appearance features
        self.hist_bins = (8, 8)
        
        # For trajectory
        self.trajectory_length = 20
        
    def _iou(self, box1, box2):
        """Calculate Intersection over Union."""
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
    
    def _center(self, bbox):
        """Get center of bounding box."""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    def _extract_features(self, frame, bbox):
        """Extract appearance features for re-identification."""
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        
        # Extract color histogram in HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, self.hist_bins, [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        
        # Also store bounding box info for size matching
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height if height > 0 else 0
        
        return {
            'hist': hist.flatten(),
            'size': (width, height),
            'aspect_ratio': aspect_ratio
        }
    
    def _compare_features(self, feat1, feat2):
        """Compare two feature sets."""
        if feat1 is None or feat2 is None:
            return 0.5
        
        # Compare histograms
        hist1 = feat1['hist'].reshape(self.hist_bins)
        hist2 = feat2['hist'].reshape(self.hist_bins)
        hist_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        hist_sim = max(0, hist_sim)  # Ensure non-negative
        
        # Compare size (normalized difference)
        size1 = feat1['size']
        size2 = feat2['size']
        size_diff = abs(size1[0] - size2[0]) / max(size1[0], size2[0]) + \
                   abs(size1[1] - size2[1]) / max(size1[1], size2[1])
        size_sim = 1.0 - min(size_diff / 2.0, 1.0)
        
        # Compare aspect ratio
        ar_sim = 1.0 - min(abs(feat1['aspect_ratio'] - feat2['aspect_ratio']), 1.0)
        
        # Combined similarity
        return (0.6 * hist_sim + 0.3 * size_sim + 0.1 * ar_sim)
    
    def register(self, frame, bbox, confidence=0.9):
        """Register a new track."""
        track_id = self.next_id
        self.next_id += 1
        
        center = self._center(bbox)
        
        self.tracks[track_id] = {
            'bbox': bbox,
            'disappeared': 0,
            'confidence': confidence,
            'trajectory': deque([center], maxlen=self.trajectory_length),
            'first_seen': time.time(),
            'last_seen': time.time(),
            'features': self._extract_features(frame, bbox) if self.use_appearance else None,
            'features_history': [],
            'active': True
        }
        
        return track_id
    
    def update_track(self, track_id, frame, bbox, confidence=0.9):
        """Update an existing track."""
        if track_id not in self.tracks:
            return False
        
        track = self.tracks[track_id]
        track['bbox'] = bbox
        track['disappeared'] = 0
        track['confidence'] = confidence
        track['last_seen'] = time.time()
        
        # Update trajectory
        center = self._center(bbox)
        track['trajectory'].append(center)
        
        # Update appearance features (with learning)
        if self.use_appearance and frame is not None:
            new_features = self._extract_features(frame, bbox)
            if new_features is not None:
                # Store in history
                track['features_history'].append(new_features)
                if len(track['features_history']) > 10:
                    track['features_history'].pop(0)
                
                # Update main features as average of recent history
                if track['features_history']:
                    # Simple averaging of histograms
                    avg_hist = np.mean([f['hist'] for f in track['features_history']], axis=0)
                    avg_size = np.mean([f['size'] for f in track['features_history']], axis=0)
                    avg_ar = np.mean([f['aspect_ratio'] for f in track['features_history']])
                    
                    track['features'] = {
                        'hist': avg_hist,
                        'size': tuple(avg_size),
                        'aspect_ratio': avg_ar
                    }
        
        return True
    
    def _predict_position(self, track):
        """Simple position prediction based on trajectory."""
        if len(track['trajectory']) < 2:
            return track['bbox']
        
        traj = list(track['trajectory'])
        dx = traj[-1][0] - traj[-2][0]
        dy = traj[-1][1] - traj[-2][1]
        
        current = track['bbox']
        return [
            current[0] + dx,
            current[1] + dy,
            current[2] + dx,
            current[3] + dy
        ]
    
    def update(self, frame, detections, confidences=None):
        """
        Main update function with re-identification.
        
        Args:
            frame: Current frame
            detections: List of [x1, y1, x2, y2]
            confidences: List of confidence scores
            
        Returns:
            Dictionary of active tracks
        """
        if confidences is None:
            confidences = [0.9] * len(detections)
        
        # STEP 1: Mark all existing tracks as disappeared initially
        for track_id in self.tracks:
            self.tracks[track_id]['disappeared'] += 1
        
        # If no detections, just clean up old tracks
        if not detections:
            self._cleanup()
            return self.get_active_tracks()
        
        # STEP 2: Match detections to existing tracks
        matches = []
        used_detections = set()
        
        # First pass: IoU-based matching for high confidence matches
        for i, det in enumerate(detections):
            best_iou = 0
            best_track = None
            
            for track_id, track in self.tracks.items():
                # Only consider recently seen tracks
                if track['disappeared'] > 5:
                    continue
                    
                predicted_bbox = track['bbox'] if track['disappeared'] == 1 else self._predict_position(track)
                iou = self._iou(predicted_bbox, det)
                
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_track = track_id
            
            if best_track is not None:
                matches.append((best_track, i))
                used_detections.add(i)
                self.update_track(best_track, frame, det, confidences[i] if i < len(confidences) else 0.9)
        
        # STEP 3: Re-identification for unmatched tracks
        if self.use_appearance and len(detections) > 0:
            for track_id, track in self.tracks.items():
                # Skip if already matched or recently disappeared
                if track['disappeared'] < 2 or any(track_id == m[0] for m in matches):
                    continue
                
                # Try to re-identify based on appearance
                best_similarity = 0
                best_det_idx = -1
                
                for i, det in enumerate(detections):
                    if i in used_detections:
                        continue
                    
                    # Extract features for this detection
                    det_features = self._extract_features(frame, det)
                    if det_features is None or track['features'] is None:
                        continue
                    
                    # Compare features
                    similarity = self._compare_features(track['features'], det_features)
                    
                    # Also check if sizes are compatible
                    det_width = det[2] - det[0]
                    det_height = det[3] - det[1]
                    track_width = track['bbox'][2] - track['bbox'][0]
                    track_height = track['bbox'][3] - track['bbox'][1]
                    
                    size_ratio_w = min(det_width, track_width) / max(det_width, track_width)
                    size_ratio_h = min(det_height, track_height) / max(det_height, track_height)
                    
                    # Combined score with size constraint
                    combined_score = (0.7 * similarity + 0.3 * (size_ratio_w + size_ratio_h) / 2)
                    
                    if combined_score > 0.6 and combined_score > best_similarity:
                        best_similarity = combined_score
                        best_det_idx = i
                
                # If good match found, re-identify
                if best_det_idx >= 0 and best_similarity > 0.6:
                    matches.append((track_id, best_det_idx))
                    used_detections.add(best_det_idx)
                    self.update_track(track_id, frame, detections[best_det_idx], 
                                     confidences[best_det_idx] if best_det_idx < len(confidences) else 0.9)
        
        # STEP 4: Register new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i not in used_detections:
                self.register(frame, det, confidences[i] if i < len(confidences) else 0.9)
        
        # STEP 5: Clean up old tracks
        self._cleanup()
        
        return self.get_active_tracks()
    
    def _cleanup(self):
        """Remove tracks that have disappeared for too long."""
        to_remove = []
        for track_id, track in self.tracks.items():
            if track['disappeared'] > self.max_disappeared:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
    
    def get_active_tracks(self):
        """Get currently active tracks."""
        active = {}
        for track_id, track in self.tracks.items():
            if track['disappeared'] == 0:  # Currently visible
                active[track_id] = track.copy()
        return active