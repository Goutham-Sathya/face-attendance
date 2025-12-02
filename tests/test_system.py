"""
Complete System Tests
Tests for all modules in the Face Attendance System
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_imports():
    """Test that all required modules can be imported."""
    print("\n" + "="*60)
    print("TEST: Module Imports")
    print("="*60)
    
    modules_to_test = [
        ('mediapipe', 'MediaPipe'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('sklearn', 'scikit-learn'),
        ('joblib', 'Joblib'),
    ]
    
    all_passed = True
    for module_name, display_name in modules_to_test:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {display_name}: {version}")
        except ImportError as e:
            print(f"✗ {display_name}: FAILED - {e}")
            all_passed = False
    
    return all_passed


def test_project_structure():
    """Test that required directories exist."""
    print("\n" + "="*60)
    print("TEST: Project Structure")
    print("="*60)
    
    required_dirs = [
        'src',
        'src/detection',
        'src/capture',
        'src/preprocessing',
        'src/training',
        'src/inference',
        'src/app',
        'data',
        'data/raw',
        'data/processed',
        'data/attendance_logs',
        'models',
        'tests',
        'logs',
    ]
    
    all_passed = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} - MISSING")
            all_passed = False
    
    return all_passed


def test_detection_module():
    """Test detection module functionality."""
    print("\n" + "="*60)
    print("TEST: Detection Module")
    print("="*60)
    
    try:
        from detection.mediapipe_detector import MediaPipeFaceDetector, detect_faces, get_largest_face
        
        # Test class instantiation
        detector = MediaPipeFaceDetector()
        print("✓ MediaPipeFaceDetector instantiated")
        
        # Test methods exist
        assert hasattr(detector, 'detect_faces'), "detect_faces method missing"
        assert hasattr(detector, 'get_largest_face_bbox'), "get_largest_face_bbox method missing"
        assert hasattr(detector, 'crop_face'), "crop_face method missing"
        assert hasattr(detector, 'draw_detections'), "draw_detections method missing"
        print("✓ All required methods exist")
        
        # Test convenience functions
        assert callable(detect_faces), "detect_faces function not callable"
        assert callable(get_largest_face), "get_largest_face function not callable"
        print("✓ Convenience functions available")
        
        detector.close()
        return True
        
    except Exception as e:
        print(f"✗ Detection module test failed: {e}")
        return False


def test_capture_module():
    """Test capture module imports."""
    print("\n" + "="*60)
    print("TEST: Capture Module")
    print("="*60)
    
    try:
        from capture.capture_dataset import DatasetCapturer
        
        # Test class instantiation
        capturer = DatasetCapturer()
        print("✓ DatasetCapturer instantiated")
        
        # Test methods exist
        assert hasattr(capturer, 'setup_camera'), "setup_camera method missing"
        assert hasattr(capturer, 'capture_dataset'), "capture_dataset method missing"
        print("✓ All required methods exist")
        
        return True
        
    except Exception as e:
        print(f"✗ Capture module test failed: {e}")
        return False


def test_preprocessing_module():
    """Test preprocessing module."""
    print("\n" + "="*60)
    print("TEST: Preprocessing Module")
    print("="*60)
    
    try:
        from preprocessing.preprocess import ImagePreprocessor
        import cv2
        
        # Test class instantiation
        preprocessor = ImagePreprocessor()
        print("✓ ImagePreprocessor instantiated")
        
        # Test methods exist
        assert hasattr(preprocessor, 'preprocess_image'), "preprocess_image method missing"
        assert hasattr(preprocessor, 'preprocess_file'), "preprocess_file method missing"
        assert hasattr(preprocessor, 'preprocess_dataset'), "preprocess_dataset method missing"
        print("✓ All required methods exist")
        
        # Test preprocessing on dummy image
        dummy_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        processed = preprocessor.preprocess_image(dummy_image)
        assert processed is not None, "Preprocessing returned None"
        assert processed.shape == (96, 96), f"Wrong output shape: {processed.shape}"
        print("✓ Image preprocessing works")
        
        return True
        
    except Exception as e:
        print(f"✗ Preprocessing module test failed: {e}")
        return False


def test_training_module():
    """Test training module imports."""
    print("\n" + "="*60)
    print("TEST: Training Module")
    print("="*60)
    
    try:
        from training.train_model import FaceRecognitionTrainer
        
        # Test class instantiation
        trainer = FaceRecognitionTrainer()
        print("✓ FaceRecognitionTrainer instantiated")
        
        # Test methods exist
        assert hasattr(trainer, 'load_dataset'), "load_dataset method missing"
        assert hasattr(trainer, 'train'), "train method missing"
        assert hasattr(trainer, 'save_models'), "save_models method missing"
        print("✓ All required methods exist")
        
        return True
        
    except Exception as e:
        print(f"✗ Training module test failed: {e}")
        return False


def test_inference_module():
    """Test inference module imports."""
    print("\n" + "="*60)
    print("TEST: Inference Module")
    print("="*60)
    
    try:
        from inference.run_inference import AttendanceSystem
        
        print("✓ AttendanceSystem imported (instantiation requires trained models)")
        
        return True
        
    except Exception as e:
        print(f"✗ Inference module test failed: {e}")
        return False


def test_cli_module():
    """Test CLI module imports."""
    print("\n" + "="*60)
    print("TEST: CLI Module")
    print("="*60)
    
    try:
        from app import main
        
        print("✓ CLI main module imported")
        
        # Test that main functions exist
        assert hasattr(main, 'main'), "main function missing"
        print("✓ Main function exists")
        
        return True
        
    except Exception as e:
        print(f"✗ CLI module test failed: {e}")
        return False


def test_file_structure():
    """Test that all required Python files exist."""
    print("\n" + "="*60)
    print("TEST: Python Files")
    print("="*60)
    
    required_files = [
        'src/__init__.py',
        'src/detection/__init__.py',
        'src/detection/mediapipe_detector.py',
        'src/capture/__init__.py',
        'src/capture/capture_dataset.py',
        'src/preprocessing/__init__.py',
        'src/preprocessing/preprocess.py',
        'src/training/__init__.py',
        'src/training/train_model.py',
        'src/inference/__init__.py',
        'src/inference/run_inference.py',
        'src/app/__init__.py',
        'src/app/main.py',
        'requirements.txt',
        'README.md',
        '.gitignore',
    ]
    
    all_passed = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            all_passed = False
    
    return all_passed


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*70)
    print(" "*20 + "FACE ATTENDANCE SYSTEM")
    print(" "*25 + "TEST SUITE")
    print("="*70)
    
    tests = [
        ("Module Imports", test_imports),
        ("Project Structure", test_project_structure),
        ("Python Files", test_file_structure),
        ("Detection Module", test_detection_module),
        ("Capture Module", test_capture_module),
        ("Preprocessing Module", test_preprocessing_module),
        ("Training Module", test_training_module),
        ("Inference Module", test_inference_module),
        ("CLI Module", test_cli_module),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ {test_name} CRASHED: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status:12} {test_name}")
    
    print("="*70)
    print(f"Total: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED! System is ready to use.")
        print("\nNext steps:")
        print("  1. python src\\app\\main.py capture \"Student Name\" --count 10")
        print("  2. python src\\app\\main.py preprocess")
        print("  3. python src\\app\\main.py train")
        print("  4. python src\\app\\main.py run")
    else:
        print(f"\n⚠ {total_count - passed_count} test(s) failed. Please fix issues above.")
    
    print("="*70 + "\n")
    
    return passed_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)