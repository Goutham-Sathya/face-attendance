'''mediapipe_detector.py'''

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List


class MediaPipeFaceDetector:
    """
    Wrapper class for MediaPipe face detection.
    Handles face detection, bounding box extraction, and face cropping.
    """
    
    def __init__(self, min_detection_confidence: float = 0.5):
        """
        Initialize MediaPipe face detector.
        
        Args:
            min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for face detection
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.detector = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence
        )
        
    def detect_faces(self, image: np.ndarray) -> Optional[List]:
        """
        Detect all faces in an image.
        
        Args:
            image: BGR image (OpenCV format)
            
        Returns:
            List of detection objects or None if no faces found
        """
        # Convert BGR to RGB (MediaPipe expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.detector.process(image_rgb)
        
        if results.detections:
            return results.detections
        return None
    
    def get_largest_face_bbox(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Get bounding box of the largest face in the image.
        
        Args:
            image: BGR image (OpenCV format)
            
        Returns:
            Tuple of (x, y, width, height) or None if no face found
        """
        detections = self.detect_faces(image)
        
        if not detections:
            return None
        
        h, w, _ = image.shape
        largest_bbox = None
        largest_area = 0
        
        for detection in detections:
            bbox = detection.location_data.relative_bounding_box
            
            # Convert relative coordinates to absolute pixels
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            area = width * height
            
            if area > largest_area:
                largest_area = area
                largest_bbox = (x, y, width, height)
        
        return largest_bbox
    
    def crop_face(self, image: np.ndarray, margin: float = 0.2) -> Optional[np.ndarray]:
        """
        Detect and crop the largest face from an image with optional margin.
        
        Args:
            image: BGR image (OpenCV format)
            margin: Fraction of face size to add as margin (default 0.2 = 20%)
            
        Returns:
            Cropped face image or None if no face found
        """
        bbox = self.get_largest_face_bbox(image)
        
        if bbox is None:
            return None
        
        x, y, width, height = bbox
        h, w, _ = image.shape
        
        # Add margin
        margin_x = int(width * margin)
        margin_y = int(height * margin)
        
        # Calculate new coordinates with margin
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w, x + width + margin_x)
        y2 = min(h, y + height + margin_y)
        
        # Crop the face
        face_crop = image[y1:y2, x1:x2]
        
        return face_crop
    
    def draw_detections(self, image: np.ndarray, draw_landmarks: bool = False) -> np.ndarray:
        """
        Draw bounding boxes around detected faces on the image.
        
        Args:
            image: BGR image (OpenCV format)
            draw_landmarks: Whether to draw facial landmarks (default False)
            
        Returns:
            Image with drawn detections
        """
        image_copy = image.copy()
        detections = self.detect_faces(image)
        
        if not detections:
            return image_copy
        
        h, w, _ = image.shape
        
        for detection in detections:
            bbox = detection.location_data.relative_bounding_box
            
            # Convert to absolute coordinates
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Draw rectangle
            cv2.rectangle(image_copy, (x, y), (x + width, y + height), (0, 255, 0), 2)
            
            # Draw confidence score
            confidence = detection.score[0]
            cv2.putText(
                image_copy, 
                f'{confidence:.2f}', 
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                2
            )
        
        return image_copy
    
    def close(self):
        """Release MediaPipe resources."""
        self.detector.close()


# Convenience functions for quick use
def detect_faces(image: np.ndarray, confidence: float = 0.5) -> Optional[List]:
    """
    Quick function to detect faces without creating detector instance.
    
    Args:
        image: BGR image (OpenCV format)
        confidence: Minimum detection confidence
        
    Returns:
        List of detections or None
    """
    detector = MediaPipeFaceDetector(min_detection_confidence=confidence)
    detections = detector.detect_faces(image)
    detector.close()
    return detections


def get_largest_face(image: np.ndarray, margin: float = 0.2, confidence: float = 0.5) -> Optional[np.ndarray]:
    """
    Quick function to get cropped largest face from image.
    
    Args:
        image: BGR image (OpenCV format)
        margin: Margin around face (fraction)
        confidence: Minimum detection confidence
        
    Returns:
        Cropped face image or None
    """
    detector = MediaPipeFaceDetector(min_detection_confidence=confidence)
    face_crop = detector.crop_face(image, margin=margin)
    detector.close()
    return face_crop


if __name__ == "__main__":
    """
    Test the detector with webcam.
    Run: python src/detection/mediapipe_detector.py
    Press 'q' to quit.
    """
    print("Testing MediaPipe Face Detector...")
    print("Press 'q' to quit")
    
    # Initialize detector
    detector = MediaPipeFaceDetector(min_detection_confidence=0.5)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        exit()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Draw detections on frame
        annotated_frame = detector.draw_detections(frame)
        
        # Show the frame
        cv2.imshow('MediaPipe Face Detection Test', annotated_frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    
    print("Test complete!")