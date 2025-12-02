"""
Live Inference Module
Real-time face recognition and attendance logging
"""

import cv2
import numpy as np
import joblib
import argparse
import csv
from pathlib import Path
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from detection.mediapipe_detector import MediaPipeFaceDetector


class AttendanceSystem:
    """Real-time face recognition and attendance tracking."""
    
    def __init__(self,
                 models_dir: Path,
                 attendance_dir: Path,
                 confidence_threshold: float = 0.6,
                 camera_index: int = 0,
                 target_size: tuple = (96, 96)):
        """
        Initialize attendance system.
        
        Args:
            models_dir: Directory containing trained models
            attendance_dir: Directory to save attendance logs
            confidence_threshold: Minimum confidence for recognition
            camera_index: Webcam device index
            target_size: Image size for preprocessing (must match training)
        """
        self.models_dir = Path(models_dir)
        self.attendance_dir = Path(attendance_dir)
        self.confidence_threshold = confidence_threshold
        self.camera_index = camera_index
        self.target_size = target_size
        
        # Load models
        self.load_models()
        
        # Initialize detector
        self.detector = MediaPipeFaceDetector(min_detection_confidence=0.5)
        
        # Track who has been marked present today
        self.marked_today = set()
        
        # Attendance log file for today
        self.attendance_file = self.get_todays_attendance_file()
        
        # Load existing attendance for today
        self.load_todays_attendance()
    
    def load_models(self):
        """Load trained PCA and KNN models."""
        print(f"Loading models from {self.models_dir}...")
        
        pca_path = self.models_dir / 'pca.joblib'
        knn_path = self.models_dir / 'knn.joblib'
        encoders_path = self.models_dir / 'label_encoders.joblib'
        
        if not pca_path.exists():
            raise FileNotFoundError(f"PCA model not found: {pca_path}")
        if not knn_path.exists():
            raise FileNotFoundError(f"KNN model not found: {knn_path}")
        if not encoders_path.exists():
            raise FileNotFoundError(f"Label encoders not found: {encoders_path}")
        
        self.pca = joblib.load(pca_path)
        self.knn = joblib.load(knn_path)
        encoders = joblib.load(encoders_path)
        self.label_decoder = encoders['decoder']
        
        print(f"✓ Models loaded successfully")
        print(f"  Known persons: {list(self.label_decoder.values())}")
    
    def get_todays_attendance_file(self) -> Path:
        """Get path to today's attendance CSV file."""
        self.attendance_dir.mkdir(parents=True, exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        return self.attendance_dir / f"attendance_{today}.csv"
    
    def load_todays_attendance(self):
        """Load names already marked present today."""
        if self.attendance_file.exists():
            with open(self.attendance_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.marked_today.add(row['name'])
            print(f"✓ Loaded existing attendance: {len(self.marked_today)} person(s) already marked")
    
    def mark_attendance(self, name: str):
        """
        Mark a person as present in the attendance log.
        
        Args:
            name: Person's name
        """
        if name in self.marked_today:
            return  # Already marked today
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Write to CSV
        file_exists = self.attendance_file.exists()
        with open(self.attendance_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['name', 'timestamp'])
            if not file_exists:
                writer.writeheader()
            writer.writerow({'name': name, 'timestamp': timestamp})
        
        self.marked_today.add(name)
        print(f"✓ ATTENDANCE MARKED: {name} at {timestamp}")
    
    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess face image to match training format.
        
        Args:
            face_image: Cropped face image
            
        Returns:
            Preprocessed feature vector
        """
        # Convert to grayscale if needed
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
        
        # Resize to target size
        resized = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Histogram equalization
        equalized = cv2.equalizeHist(resized)
        
        # Flatten and normalize
        feature_vector = equalized.flatten() / 255.0
        
        return feature_vector.reshape(1, -1)
    
    def recognize_face(self, face_image: np.ndarray) -> tuple:
        """
        Recognize a face.
        
        Args:
            face_image: Cropped face image
            
        Returns:
            Tuple of (name, confidence)
        """
        # Preprocess
        features = self.preprocess_face(face_image)
        
        # Apply PCA
        features_pca = self.pca.transform(features)
        
        # Predict with KNN
        prediction = self.knn.predict(features_pca)[0]
        probabilities = self.knn.predict_proba(features_pca)[0]
        confidence = probabilities[prediction]
        
        # Decode label
        name = self.label_decoder[prediction]
        
        return name, confidence
    
    def run(self):
        """Run live attendance system."""
        print(f"\n{'='*60}")
        print("STARTING ATTENDANCE SYSTEM")
        print(f"{'='*60}")
        print(f"Attendance file: {self.attendance_file}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print(f"Already marked today: {len(self.marked_today)} person(s)")
        print(f"\nPress 'q' to quit")
        print(f"Press 'r' to reset today's attendance")
        print(f"{'='*60}\n")
        
        # Open camera
        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            print(f"ERROR: Could not open camera with index {self.camera_index}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        frame_count = 0
        recognition_interval = 10  # Recognize every 10 frames
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("ERROR: Failed to capture frame")
                break
            
            frame_count += 1
            display_frame = frame.copy()
            
            # Detect face
            face_crop = self.detector.crop_face(frame, margin=0.2)
            bbox = self.detector.get_largest_face_bbox(frame)
            
            if face_crop is not None and bbox is not None:
                x, y, w, h = bbox
                
                # Recognize face periodically
                if frame_count % recognition_interval == 0:
                    name, confidence = self.recognize_face(face_crop)
                    
                    # Store for display
                    self.current_name = name
                    self.current_confidence = confidence
                    
                    # Mark attendance if confident enough
                    if confidence >= self.confidence_threshold:
                        self.mark_attendance(name)
                
                # Draw bounding box
                if hasattr(self, 'current_confidence'):
                    if self.current_confidence >= self.confidence_threshold:
                        color = (0, 255, 0)  # Green for recognized
                        status = "PRESENT"
                    else:
                        color = (0, 165, 255)  # Orange for low confidence
                        status = "UNCERTAIN"
                else:
                    color = (255, 255, 0)  # Yellow for detecting
                    status = "DETECTING"
                
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                
                # Display name and confidence
                if hasattr(self, 'current_name'):
                    label = f"{self.current_name} ({self.current_confidence:.0%})"
                    cv2.putText(display_frame, label, (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Display status
                    cv2.putText(display_frame, status, (x, y + h + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Display attendance count
            cv2.putText(display_frame, f"Present Today: {len(self.marked_today)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display instructions
            cv2.putText(display_frame, "Q: Quit | R: Reset", 
                       (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('Attendance System', display_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nShutting down...")
                break
            elif key == ord('r'):
                # Reset attendance for today
                response = input("\nReset today's attendance? (yes/no): ")
                if response.lower() == 'yes':
                    self.marked_today.clear()
                    if self.attendance_file.exists():
                        self.attendance_file.unlink()
                    print("✓ Attendance reset")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.detector.close()
        
        print(f"\n{'='*60}")
        print("ATTENDANCE SYSTEM STOPPED")
        print(f"{'='*60}")
        print(f"Total marked today: {len(self.marked_today)}")
        print(f"Names: {sorted(self.marked_today)}")
        print(f"Attendance log: {self.attendance_file}")
        print(f"{'='*60}\n")


def main():
    """Command-line interface for attendance system."""
    parser = argparse.ArgumentParser(
        description="Run live face recognition attendance system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python run_inference.py
  
  # Custom confidence threshold
  python run_inference.py --threshold 0.7
  
  # Use different camera
  python run_inference.py --camera 1
        """
    )
    
    parser.add_argument('--models', '-m', type=str, default='models',
                       help='Directory containing trained models (default: models)')
    parser.add_argument('--attendance', '-a', type=str, default='data/attendance_logs',
                       help='Directory for attendance logs (default: data/attendance_logs)')
    parser.add_argument('--threshold', '-t', type=float, default=0.6,
                       help='Confidence threshold for recognition (default: 0.6)')
    parser.add_argument('--camera', '-c', type=int, default=0,
                       help='Camera device index (default: 0)')
    parser.add_argument('--size', '-s', type=int, default=96,
                       help='Image size for preprocessing (default: 96, must match training)')
    
    args = parser.parse_args()
    
    try:
        # Create system
        system = AttendanceSystem(
            models_dir=Path(args.models),
            attendance_dir=Path(args.attendance),
            confidence_threshold=args.threshold,
            camera_index=args.camera,
            target_size=(args.size, args.size)
        )
        
        # Run
        system.run()
        
        return 0
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())