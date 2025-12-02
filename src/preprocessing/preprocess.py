"""
Preprocessing Module
Converts raw captured images into normalized, fixed-size grayscale faces
"""

import cv2
import os
import sys
import argparse
from pathlib import Path
from typing import Tuple, Optional
import numpy as np


class ImagePreprocessor:
    """Preprocesses face images for feature extraction and training."""
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (96, 96),
                 apply_histogram_eq: bool = True):
        """
        Initialize preprocessor.
        
        Args:
            target_size: Target image size (width, height)
            apply_histogram_eq: Whether to apply histogram equalization
        """
        self.target_size = target_size
        self.apply_histogram_eq = apply_histogram_eq
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess a single image.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize to target size
        resized = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Apply histogram equalization for better contrast
        if self.apply_histogram_eq:
            resized = cv2.equalizeHist(resized)
        
        return resized
    
    def preprocess_file(self, 
                       input_path: Path, 
                       output_path: Path) -> bool:
        """
        Preprocess a single image file.
        
        Args:
            input_path: Path to input image
            output_path: Path to save preprocessed image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read image
            image = cv2.imread(str(input_path))
            if image is None:
                print(f"  ✗ Failed to read: {input_path.name}")
                return False
            
            # Preprocess
            processed = self.preprocess_image(image)
            
            # Save
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), processed)
            
            return True
            
        except Exception as e:
            print(f"  ✗ Error processing {input_path.name}: {e}")
            return False
    
    def preprocess_dataset(self, 
                          input_dir: Path, 
                          output_dir: Path) -> dict:
        """
        Preprocess all images in dataset.
        
        Args:
            input_dir: Directory containing raw images (organized by person)
            output_dir: Directory to save preprocessed images
            
        Returns:
            Dictionary with processing statistics
        """
        stats = {
            'total_persons': 0,
            'total_images': 0,
            'successful': 0,
            'failed': 0,
            'persons': {}
        }
        
        print(f"\n{'='*60}")
        print(f"PREPROCESSING DATASET")
        print(f"{'='*60}")
        print(f"Input:  {input_dir}")
        print(f"Output: {output_dir}")
        print(f"Target size: {self.target_size}")
        print(f"Histogram equalization: {self.apply_histogram_eq}")
        print(f"{'='*60}\n")
        
        # Check if input directory exists
        if not input_dir.exists():
            print(f"ERROR: Input directory does not exist: {input_dir}")
            return stats
        
        # Find all person directories
        person_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
        
        if not person_dirs:
            print(f"WARNING: No person directories found in {input_dir}")
            return stats
        
        stats['total_persons'] = len(person_dirs)
        
        # Process each person's images
        for person_dir in sorted(person_dirs):
            person_name = person_dir.name
            print(f"Processing: {person_name}")
            
            # Find all image files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(person_dir.glob(ext))
            
            if not image_files:
                print(f"  ⚠ No images found for {person_name}")
                continue
            
            person_stats = {
                'total': len(image_files),
                'successful': 0,
                'failed': 0
            }
            
            # Create output directory for this person
            person_output_dir = output_dir / person_name
            
            # Process each image
            for img_path in sorted(image_files):
                # Create output path with .png extension
                output_path = person_output_dir / f"{img_path.stem}.png"
                
                if self.preprocess_file(img_path, output_path):
                    person_stats['successful'] += 1
                    stats['successful'] += 1
                else:
                    person_stats['failed'] += 1
                    stats['failed'] += 1
                
                stats['total_images'] += 1
            
            # Store person stats
            stats['persons'][person_name] = person_stats
            
            print(f"  ✓ Processed {person_stats['successful']}/{person_stats['total']} images")
            if person_stats['failed'] > 0:
                print(f"  ✗ Failed: {person_stats['failed']}")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"PREPROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Persons processed: {stats['total_persons']}")
        print(f"Total images: {stats['total_images']}")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        print(f"Success rate: {stats['successful']/stats['total_images']*100:.1f}%" if stats['total_images'] > 0 else "N/A")
        print(f"{'='*60}\n")
        
        return stats
    
    def validate_preprocessed(self, output_dir: Path) -> bool:
        """
        Validate that preprocessed images have consistent sizes.
        
        Args:
            output_dir: Directory containing preprocessed images
            
        Returns:
            True if validation passes, False otherwise
        """
        print(f"\n{'='*60}")
        print(f"VALIDATING PREPROCESSED IMAGES")
        print(f"{'='*60}\n")
        
        if not output_dir.exists():
            print(f"ERROR: Output directory does not exist: {output_dir}")
            return False
        
        person_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
        
        if not person_dirs:
            print(f"WARNING: No person directories found")
            return False
        
        all_valid = True
        
        for person_dir in sorted(person_dirs):
            person_name = person_dir.name
            image_files = list(person_dir.glob('*.png'))
            
            if not image_files:
                print(f"⚠ {person_name}: No images found")
                continue
            
            # Check first image
            first_img = cv2.imread(str(image_files[0]), cv2.IMREAD_GRAYSCALE)
            if first_img is None:
                print(f"✗ {person_name}: Failed to read images")
                all_valid = False
                continue
            
            expected_shape = first_img.shape
            
            # Verify all images have same size
            invalid_count = 0
            for img_path in image_files:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None or img.shape != expected_shape:
                    invalid_count += 1
            
            if invalid_count > 0:
                print(f"✗ {person_name}: {invalid_count}/{len(image_files)} images have incorrect size")
                all_valid = False
            else:
                print(f"✓ {person_name}: {len(image_files)} images, size {expected_shape}")
        
        print(f"\n{'='*60}")
        if all_valid:
            print("✓ VALIDATION PASSED")
        else:
            print("✗ VALIDATION FAILED")
        print(f"{'='*60}\n")
        
        return all_valid


def main():
    """Command-line interface for preprocessing."""
    parser = argparse.ArgumentParser(
        description="Preprocess face images for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess with default settings (96x96, histogram equalization)
  python preprocess.py
  
  # Custom size and disable histogram equalization
  python preprocess.py --size 128 --no-histogram
  
  # Custom input/output directories
  python preprocess.py --input data/raw --output data/processed
        """
    )
    
    parser.add_argument('--input', '-i', type=str, default='data/raw',
                       help='Input directory with raw images (default: data/raw)')
    parser.add_argument('--output', '-o', type=str, default='data/processed',
                       help='Output directory for processed images (default: data/processed)')
    parser.add_argument('--size', '-s', type=int, default=96,
                       help='Target image size (width=height) (default: 96)')
    parser.add_argument('--no-histogram', action='store_true',
                       help='Disable histogram equalization')
    parser.add_argument('--validate', '-v', action='store_true',
                       help='Only validate existing preprocessed images')
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = ImagePreprocessor(
        target_size=(args.size, args.size),
        apply_histogram_eq=not args.no_histogram
    )
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    # Validate mode
    if args.validate:
        success = preprocessor.validate_preprocessed(output_dir)
        return 0 if success else 1
    
    # Preprocess dataset
    stats = preprocessor.preprocess_dataset(input_dir, output_dir)
    
    # Validate results
    if stats['successful'] > 0:
        preprocessor.validate_preprocessed(output_dir)
    
    return 0 if stats['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())