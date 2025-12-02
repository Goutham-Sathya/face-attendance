"""
Model Training Module
Extracts features and trains PCA + KNN classifier for face recognition
"""

import cv2
import numpy as np
import joblib
import argparse
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from datetime import datetime
import sys


class FaceRecognitionTrainer:
    """Handles feature extraction and model training."""
    
    def __init__(self, 
                 pca_components: int = 50,
                 knn_neighbors: int = 3):
        """
        Initialize trainer.
        
        Args:
            pca_components: Number of PCA components (default: 50)
            knn_neighbors: Number of neighbors for KNN (default: 3)
        """
        self.pca_components = pca_components
        self.knn_neighbors = knn_neighbors
        self.pca = None
        self.knn = None
        self.label_encoder = {}  # Map labels to indices
        self.label_decoder = {}  # Map indices back to labels
    
    def load_dataset(self, data_dir: Path) -> tuple:
        """
        Load and extract features from processed images.
        
        Args:
            data_dir: Directory containing processed images organized by person
            
        Returns:
            Tuple of (X, y) where X is feature matrix and y is labels
        """
        print(f"\n{'='*60}")
        print("LOADING DATASET")
        print(f"{'='*60}")
        print(f"Data directory: {data_dir}")
        
        data_dir = Path(data_dir)
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Get all person directories
        person_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        
        if not person_dirs:
            raise ValueError(f"No person directories found in {data_dir}")
        
        print(f"Found {len(person_dirs)} person(s)")
        
        X = []  # Features
        y = []  # Labels
        
        for person_dir in person_dirs:
            person_name = person_dir.name
            
            # Get all images for this person
            image_files = list(person_dir.glob("*.png")) + \
                         list(person_dir.glob("*.jpg")) + \
                         list(person_dir.glob("*.jpeg"))
            
            if not image_files:
                print(f"  ⚠ No images found for {person_name}, skipping...")
                continue
            
            print(f"  Loading {len(image_files)} images for {person_name}...")
            
            for img_path in image_files:
                # Read image (should already be grayscale)
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    print(f"    ✗ Failed to load: {img_path.name}")
                    continue
                
                # Flatten image to 1D vector
                feature_vector = img.flatten()
                
                # Normalize to [0, 1]
                feature_vector = feature_vector / 255.0
                
                X.append(feature_vector)
                y.append(person_name)
        
        if len(X) == 0:
            raise ValueError("No valid images loaded!")
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Create label encoders
        unique_labels = np.unique(y)
        self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        self.label_decoder = {idx: label for label, idx in self.label_encoder.items()}
        
        # Encode labels to integers
        y_encoded = np.array([self.label_encoder[label] for label in y])
        
        print(f"\nDataset loaded:")
        print(f"  Total samples: {len(X)}")
        print(f"  Feature dimension: {X.shape[1]}")
        print(f"  Number of classes: {len(unique_labels)}")
        print(f"  Classes: {list(unique_labels)}")
        
        # Show distribution
        print(f"\nClass distribution:")
        for label in unique_labels:
            count = np.sum(y == label)
            print(f"  {label}: {count} images")
        
        print(f"{'='*60}\n")
        
        return X, y_encoded
    
    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Train PCA + KNN model.
        
        Args:
            X: Feature matrix (samples x features)
            y: Labels (encoded as integers)
            
        Returns:
            Dictionary with training statistics
        """
        print(f"{'='*60}")
        print("TRAINING MODEL")
        print(f"{'='*60}")
        
        # Adjust PCA components if needed
        max_components = min(X.shape[0], X.shape[1])
        if self.pca_components > max_components:
            print(f"⚠ Adjusting PCA components from {self.pca_components} to {max_components}")
            self.pca_components = max_components - 1
        
        print(f"PCA components: {self.pca_components}")
        print(f"KNN neighbors: {self.knn_neighbors}")
        
        # Train PCA
        print("\nTraining PCA...")
        self.pca = PCA(n_components=self.pca_components)
        X_pca = self.pca.fit_transform(X)
        
        explained_variance = np.sum(self.pca.explained_variance_ratio_) * 100
        print(f"✓ PCA trained")
        print(f"  Explained variance: {explained_variance:.2f}%")
        print(f"  Reduced dimension: {X_pca.shape[1]}")
        
        # Train KNN
        print("\nTraining KNN classifier...")
        self.knn = KNeighborsClassifier(n_neighbors=self.knn_neighbors)
        self.knn.fit(X_pca, y)
        print(f"✓ KNN trained")
        
        # Cross-validation
        print("\nPerforming cross-validation...")
        cv_scores = cross_val_score(self.knn, X_pca, y, cv=min(5, len(np.unique(y))))
        
        stats = {
            'pca_components': self.pca_components,
            'knn_neighbors': self.knn_neighbors,
            'explained_variance': explained_variance,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_samples': len(X),
            'n_classes': len(np.unique(y))
        }
        
        print(f"  Cross-validation accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")
        print(f"{'='*60}\n")
        
        return stats
    
    def save_models(self, output_dir: Path):
        """
        Save trained models to disk.
        
        Args:
            output_dir: Directory to save models
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"{'='*60}")
        print("SAVING MODELS")
        print(f"{'='*60}")
        
        # Save PCA
        pca_path = output_dir / 'pca.joblib'
        joblib.dump(self.pca, pca_path)
        print(f"✓ PCA saved to: {pca_path}")
        
        # Save KNN
        knn_path = output_dir / 'knn.joblib'
        joblib.dump(self.knn, knn_path)
        print(f"✓ KNN saved to: {knn_path}")
        
        # Save label encoders
        encoders_path = output_dir / 'label_encoders.joblib'
        joblib.dump({
            'encoder': self.label_encoder,
            'decoder': self.label_decoder
        }, encoders_path)
        print(f"✓ Label encoders saved to: {encoders_path}")
        
        print(f"{'='*60}\n")
    
    def save_training_report(self, stats: dict, output_dir: Path):
        """
        Save training report to text file.
        
        Args:
            stats: Training statistics dictionary
            output_dir: Directory to save report
        """
        output_dir = Path(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f'training_report_{timestamp}.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("FACE RECOGNITION TRAINING REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Model Configuration:\n")
            f.write(f"  PCA Components: {stats['pca_components']}\n")
            f.write(f"  KNN Neighbors: {stats['knn_neighbors']}\n\n")
            
            f.write("Dataset:\n")
            f.write(f"  Training Samples: {stats['training_samples']}\n")
            f.write(f"  Number of Classes: {stats['n_classes']}\n")
            f.write(f"  Classes: {list(self.label_decoder.values())}\n\n")
            
            f.write("Performance:\n")
            f.write(f"  PCA Explained Variance: {stats['explained_variance']:.2f}%\n")
            f.write(f"  Cross-Validation Accuracy: {stats['cv_mean']:.2%} (+/- {stats['cv_std']:.2%})\n\n")
            
            f.write("="*60 + "\n")
        
        print(f"✓ Training report saved to: {report_path}")


def main():
    """Command-line interface for training."""
    parser = argparse.ArgumentParser(
        description="Train face recognition model (PCA + KNN)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python train_model.py
  
  # Custom PCA components and KNN neighbors
  python train_model.py --pca-components 100 --knn-k 5
  
  # Custom data and model directories
  python train_model.py --data data/processed --models models
        """
    )
    
    parser.add_argument('--data', '-d', type=str, default='data/processed',
                       help='Directory containing preprocessed images (default: data/processed)')
    parser.add_argument('--models', '-m', type=str, default='models',
                       help='Directory to save trained models (default: models)')
    parser.add_argument('--pca-components', type=int, default=50,
                       help='Number of PCA components (default: 50)')
    parser.add_argument('--knn-k', type=int, default=3,
                       help='Number of KNN neighbors (default: 3)')
    
    args = parser.parse_args()
    
    # Create trainer
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
        print(f"Models saved to: {args.models}/")
        print(f"Report saved to: logs/\n")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())