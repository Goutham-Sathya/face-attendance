"""
Dataset Capture Module
Captures face images from webcam for training the attendance system
"""

import cv2
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from detection.mediapipe_detector import MediaPipeFaceDetector


class DatasetCapturer:
    """Captures face images from webcam for dataset creation."""
    
    def __init__(self, 
                 output_dir: str = "data/raw",
                 camera_index: int = 0,
                 min_confidence: float = 0.5):
        """
        Initialize dataset capturer.
        
        Args:
            output_dir: Base directory to save captured images
            camera_index: Webcam device index (usually 0)
            min_confidence: Minimum face detection confidence
        """
        self.output_dir = Path(output_dir)
        self.camera_index = camera_index
        self.detector = MediaPipeFaceDetector(min_detection_confidence=min_confidence)
        self.cap = None
        
    def setup_camera(self) -> bool:
        """
        Initialize webcam.
        
        Returns:
            True if camera opened successfully, False otherwise
        """
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"ERROR: Could not open camera with index {self.camera_index}")
            return False
            
        # Set camera properties for better quality
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"✓ Camera opened successfully")
        return True
    
    def capture_dataset(self, 
                       person_name: str, 
                       num_images: int = 10,
                       margin: float = 0.2) -> bool:
        """
        Capture face images for a person.
        
        Args:
            person_name: Name of the person (used for folder name)
            num_images: Number of images to capture
            margin: Margin around detected face (0.2 = 20%)
            
        Returns:
            True if capture successful, False otherwise
        """
        # Create person directory
        person_dir = self.output_dir / person_name
        person_dir.mkdir(parents=True, exist_ok=True)
        
        # Create safe filename prefix (replace spaces with underscores)
        safe_name = person_name.replace(" ", "_")
        
        print(f"\n{'='*60}")
        print(f"CAPTURING DATASET FOR: {person_name}")
        print(f"{'='*60}")
        print(f"Target images: {num_images}")
        print(f"Output directory: {person_dir}")
        print(f"\nINSTRUCTIONS:")
        print("  - Look at the camera")
        print("  - Move your head slightly between captures")
        print("  - Try different expressions")
        print("  - Press 'q' to quit early")
        print(f"{'='*60}\n")
        
        if not self.setup_camera():
            return False
        
        captured_count = 0
        frame_count = 0
        capture_interval = 15  # Capture every 15 frames (~0.5 seconds at 30fps)
        
        print("Starting capture in 3 seconds...")
        print("Get ready!")
        
        # Countdown
        for i in range(3, 0, -1):
            ret, frame = self.cap.read()
            if ret:
                countdown_frame = frame.copy()
                cv2.putText(countdown_frame, str(i), (250, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 10)
                cv2.imshow('Capture Dataset', countdown_frame)
                cv2.waitKey(1000)
        
        while captured_count < num_images:
            ret, frame = self.cap.read()
            
            if not ret:
                print("ERROR: Failed to capture frame")
                break
            
            frame_count += 1
            display_frame = frame.copy()
            
            # Detect face
            face_crop = self.detector.crop_face(frame, margin=margin)
            
            # Draw bounding box and info
            if face_crop is not None:
                # Get bbox for visualization
                bbox = self.detector.get_largest_face_bbox(frame)
                if bbox:
                    x, y, w, h = bbox
                    # Draw green box if ready to capture, yellow if waiting
                    color = (0, 255, 0) if frame_count % capture_interval == 0 else (0, 255, 255)
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                
                # Capture image at intervals
                if frame_count % capture_interval == 0:
                    # Save the cropped face
                    filename = f"{safe_name}_{captured_count:03d}.jpg"
                    filepath = person_dir / filename
                    cv2.imwrite(str(filepath), face_crop)
                    
                    captured_count += 1
                    print(f"✓ Captured {captured_count}/{num_images}: {filename}")
                    
                    # Visual feedback - flash green
                    cv2.rectangle(display_frame, (0, 0), (640, 480), (0, 255, 0), 10)
            else:
                # No face detected
                cv2.putText(display_frame, "NO FACE DETECTED", (150, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display progress
            progress_text = f"Captured: {captured_count}/{num_images}"
            cv2.putText(display_frame, progress_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(display_frame, "Press 'q' to quit", (10, 460),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('Capture Dataset', display_frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"\nCapture stopped by user at {captured_count}/{num_images} images")
                break
        
        # Cleanup
        self.cleanup()
        
        # Summary
        print(f"\n{'='*60}")
        if captured_count == num_images:
            print(f"✓ SUCCESS: Captured all {num_images} images for {person_name}")
        else:
            print(f"⚠ PARTIAL: Captured {captured_count}/{num_images} images for {person_name}")
        print(f"Saved to: {person_dir}")
        print(f"{'='*60}\n")
        
        return captured_count > 0
    
    def cleanup(self):
        """Release camera and close windows."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.detector.close()


def main():
    """Command-line interface for dataset capture."""
    parser = argparse.ArgumentParser(
        description="Capture face images for attendance system training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Capture 10 images for John Doe
  python capture_dataset.py --name "John Doe"
  
  # Capture 20 images for Jane Smith
  python capture_dataset.py --name "Jane Smith" --count 20
  
  # Use camera index 1 instead of 0
  python capture_dataset.py --name "Bob" --camera 1
        """
    )
    
    parser.add_argument('--name', '-n', type=str, required=True,
                       help='Name of the person (use quotes for names with spaces)')
    parser.add_argument('--count', '-c', type=int, default=10,
                       help='Number of images to capture (default: 10)')
    parser.add_argument('--output', '-o', type=str, default='data/raw',
                       help='Output directory (default: data/raw)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device index (default: 0)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Minimum face detection confidence (default: 0.5)')
    parser.add_argument('--margin', type=float, default=0.2,
                       help='Margin around face crop (default: 0.2 = 20%%)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.count < 1:
        print("ERROR: Count must be at least 1")
        return 1
    
    if not args.name.strip():
        print("ERROR: Name cannot be empty")
        return 1
    
    # Create capturer
    capturer = DatasetCapturer(
        output_dir=args.output,
        camera_index=args.camera,
        min_confidence=args.confidence
    )
    
    # Run capture
    success = capturer.capture_dataset(
        person_name=args.name,
        num_images=args.count,
        margin=args.margin
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())