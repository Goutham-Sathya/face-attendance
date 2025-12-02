"""
Main CLI Application
Single entry point for all face attendance system operations
"""

import sys
import argparse
from pathlib import Path

# Add src to path
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from capture.capture_dataset import DatasetCapturer
from preprocessing.preprocess import ImagePreprocessor
from training.train_model import FaceRecognitionTrainer
from inference.run_inference import AttendanceSystem


def command_capture(args):
    """Handle capture command."""
    print(f"\n🎥 CAPTURE MODE")
    
    capturer = DatasetCapturer(
        output_dir=args.output,
        camera_index=args.camera,
        min_confidence=args.confidence
    )
    
    success = capturer.capture_dataset(
        person_name=args.name,
        num_images=args.count,
        margin=args.margin
    )
    
    return 0 if success else 1


def command_preprocess(args):
    """Handle preprocess command."""
    print(f"\n⚙️ PREPROCESS MODE")
    
    preprocessor = ImagePreprocessor(
        target_size=(args.size, args.size),
        apply_equalization=not args.no_equalize
    )
    
    stats = preprocessor.preprocess_dataset(
        input_dir=Path(args.input),
        output_dir=Path(args.output)
    )
    
    if stats['total_images'] == 0:
        return 1
    elif stats['failed'] > 0:
        return 2
    else:
        return 0


def command_train(args):
    """Handle train command."""
    print(f"\n🎓 TRAINING MODE")
    
    trainer = FaceRecognitionTrainer(
        pca_components=args.pca_components,
        knn_neighbors=args.knn_k
    )
    
    try:
        # Load dataset
        X, y = trainer.load_dataset(Path(args.data))
        
        # Train model
        stats = trainer.train(X, y)
        
        # Save models
        trainer.save_models(Path(args.models))
        
        # Save report
        trainer.save_training_report(stats, Path('logs'))
        
        print("\n✓ Training complete!")
        return 0
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        return 1


def command_run(args):
    """Handle run command."""
    print(f"\n🚀 ATTENDANCE MODE")
    
    try:
        system = AttendanceSystem(
            models_dir=Path(args.models),
            attendance_dir=Path(args.attendance),
            confidence_threshold=args.threshold,
            camera_index=args.camera,
            target_size=(args.size, args.size)
        )
        
        system.run()
        return 0
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


def command_status(args):
    """Show system status."""
    print(f"\n📊 SYSTEM STATUS")
    print(f"{'='*60}")
    
    # Check directories
    dirs_to_check = {
        'Raw data': Path('data/raw'),
        'Processed data': Path('data/processed'),
        'Models': Path('models'),
        'Attendance logs': Path('data/attendance_logs'),
        'System logs': Path('logs')
    }
    
    print("\nDirectories:")
    for name, path in dirs_to_check.items():
        exists = "✓" if path.exists() else "✗"
        print(f"  {exists} {name}: {path}")
    
    # Count people in raw data
    raw_dir = Path('data/raw')
    if raw_dir.exists():
        person_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
        print(f"\nRaw dataset:")
        print(f"  People: {len(person_dirs)}")
        for person_dir in person_dirs:
            images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
            print(f"    - {person_dir.name}: {len(images)} images")
    
    # Count people in processed data
    proc_dir = Path('data/processed')
    if proc_dir.exists():
        person_dirs = [d for d in proc_dir.iterdir() if d.is_dir()]
        print(f"\nProcessed dataset:")
        print(f"  People: {len(person_dirs)}")
        for person_dir in person_dirs:
            images = list(person_dir.glob("*.png"))
            print(f"    - {person_dir.name}: {len(images)} images")
    
    # Check models
    models_dir = Path('models')
    print(f"\nTrained models:")
    if models_dir.exists():
        pca = models_dir / 'pca.joblib'
        knn = models_dir / 'knn.joblib'
        encoders = models_dir / 'label_encoders.joblib'
        
        print(f"  {'✓' if pca.exists() else '✗'} PCA model")
        print(f"  {'✓' if knn.exists() else '✗'} KNN classifier")
        print(f"  {'✓' if encoders.exists() else '✗'} Label encoders")
        
        if all([pca.exists(), knn.exists(), encoders.exists()]):
            print(f"\n  ✓ System is ready to run!")
        else:
            print(f"\n  ⚠ Models missing - please run 'train' command")
    else:
        print(f"  ✗ Models directory not found")
    
    # Check attendance logs
    attendance_dir = Path('data/attendance_logs')
    if attendance_dir.exists():
        logs = list(attendance_dir.glob("attendance_*.csv"))
        print(f"\nAttendance logs:")
        print(f"  Total days: {len(logs)}")
        if logs:
            print(f"  Latest: {sorted(logs)[-1].name}")
    
    print(f"\n{'='*60}\n")
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Face Attendance System - Automated classroom attendance using face recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WORKFLOW:
  1. Capture face images for each student
  2. Preprocess the captured images
  3. Train the recognition model
  4. Run the live attendance system

EXAMPLES:
  # Check system status
  python main.py status
  
  # Capture images for a student
  python main.py capture "John Doe" --count 10
  
  # Preprocess all captured images
  python main.py preprocess
  
  # Train the model
  python main.py train
  
  # Run live attendance
  python main.py run
  
  # Run with custom confidence threshold
  python main.py run --threshold 0.7

For more help on a specific command:
  python main.py <command> --help
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    
    # Capture command
    capture_parser = subparsers.add_parser('capture', help='Capture face images')
    capture_parser.add_argument('name', type=str, help='Person name (use quotes for spaces)')
    capture_parser.add_argument('--count', '-c', type=int, default=10,
                               help='Number of images to capture (default: 10)')
    capture_parser.add_argument('--output', '-o', type=str, default='data/raw',
                               help='Output directory (default: data/raw)')
    capture_parser.add_argument('--camera', type=int, default=0,
                               help='Camera index (default: 0)')
    capture_parser.add_argument('--confidence', type=float, default=0.5,
                               help='Detection confidence (default: 0.5)')
    capture_parser.add_argument('--margin', type=float, default=0.2,
                               help='Face crop margin (default: 0.2)')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess captured images')
    preprocess_parser.add_argument('--input', '-i', type=str, default='data/raw',
                                  help='Input directory (default: data/raw)')
    preprocess_parser.add_argument('--output', '-o', type=str, default='data/processed',
                                  help='Output directory (default: data/processed)')
    preprocess_parser.add_argument('--size', '-s', type=int, default=96,
                                  help='Target image size (default: 96)')
    preprocess_parser.add_argument('--no-equalize', action='store_true',
                                  help='Disable histogram equalization')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train recognition model')
    train_parser.add_argument('--data', '-d', type=str, default='data/processed',
                             help='Processed data directory (default: data/processed)')
    train_parser.add_argument('--models', '-m', type=str, default='models',
                             help='Models output directory (default: models)')
    train_parser.add_argument('--pca-components', type=int, default=50,
                             help='PCA components (default: 50)')
    train_parser.add_argument('--knn-k', type=int, default=3,
                             help='KNN neighbors (default: 3)')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run live attendance system')
    run_parser.add_argument('--models', '-m', type=str, default='models',
                           help='Models directory (default: models)')
    run_parser.add_argument('--attendance', '-a', type=str, default='data/attendance_logs',
                           help='Attendance logs directory (default: data/attendance_logs)')
    run_parser.add_argument('--threshold', '-t', type=float, default=0.6,
                           help='Recognition confidence threshold (default: 0.6)')
    run_parser.add_argument('--camera', '-c', type=int, default=0,
                           help='Camera index (default: 0)')
    run_parser.add_argument('--size', '-s', type=int, default=96,
                           help='Image size (default: 96, must match training)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Show help if no command provided
    if not args.command:
        parser.print_help()
        return 0
    
    # Route to appropriate command handler
    commands = {
        'status': command_status,
        'capture': command_capture,
        'preprocess': command_preprocess,
        'train': command_train,
        'run': command_run
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())