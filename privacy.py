
# privacy.py
import cv2

def blur_person(frame, x1, y1, x2, y2):
    """
    Blur a person's face/body in the frame.
    
    Args:
        frame: Input frame
        x1, y1: Top-left coordinates
        x2, y2: Bottom-right coordinates
        
    Returns:
        Frame with blurred region
    """
    # Ensure coordinates are within frame bounds
    height, width = frame.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)
    
    # Check if region is valid
    if x2 <= x1 or y2 <= y1:
        return frame
    
    # Extract region of interest
    roi = frame[y1:y2, x1:x2]
    
    if roi.size == 0:
        return frame
    
    # Calculate blur kernel size based on person size
    person_width = x2 - x1
    person_height = y2 - y1
    avg_size = (person_width + person_height) // 2
    
    # Make kernel size proportional to person size (odd number)
    kernel_size = min(99, max(31, avg_size // 10 * 2 + 1))
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 30)
    
    # Replace region with blurred version
    frame[y1:y2, x1:x2] = blurred
    
    return frame






