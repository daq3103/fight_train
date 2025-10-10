"""
Inference script cho Stage 1 model đã train
Sử dụng model từ checkpoint để predict video test.mp4
"""

import torch
import cv2
import numpy as np
import os
import argparse
from torchvision import transforms
from PIL import Image
import time

from models.complete_model import CompleteModel
from extract_i3d_features import I3DFeatureExtractor
from config import Config


class FightDetector:
    """Fight Detection Inference Class"""
    
    def __init__(self, checkpoint_path, device=None):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load config
        self.config = Config()
        
        # Initialize feature extractor
        self.feature_extractor = I3DFeatureExtractor()
        
        # Initialize model
        self.model = CompleteModel(
            backbone_type="r2plus1d_18",
            num_classes=self.config.num_classes,
            pretrained=True,
            feature_extract_mode=True,
        ).to(self.device)
        
        # Load trained weights
        self._load_checkpoint(checkpoint_path)
        
        # Set to evaluation mode
        self.model.eval()
        
        print("Fight Detector initialized successfully!")
    
    def _load_checkpoint(self, checkpoint_path):
        """Load trained model checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load model state dict
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            # Direct state dict
            self.model.load_state_dict(checkpoint)
            print("Loaded model state dict directly")
    
    def extract_video_features(self, video_path, target_clips=32):
        """Extract I3D features from video"""
        print(f"Extracting features from: {video_path}")
        
        # Use the feature extractor
        features = self.feature_extractor.extract_video_features(
            video_path, target_clips=target_clips
        )
        
        if features is None:
            raise ValueError(f"Failed to extract features from {video_path}")
        
        print(f"Extracted features shape: {features.shape}")
        return features
    
    def predict_video(self, video_path, threshold=0.5):
        """
        Predict fight detection for a video
        
        Args:
            video_path: Path to video file
            threshold: Classification threshold
            
        Returns:
            dict: Prediction results
        """
        start_time = time.time()
        
        # Extract features
        features = self.extract_video_features(video_path)
        
        # Convert to tensor and add batch dimension
        features_tensor = torch.from_numpy(features).float().unsqueeze(0)  # (1, 32, 1024)
        features_tensor = features_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(features_tensor)  # (1, 32, 2)
            
            # Get probabilities
            probs = torch.softmax(logits, dim=-1)  # (1, 32, 2)
            anomaly_scores = probs[0, :, 1]  # (32,) - fight probabilities
            
            # Video-level prediction (max pooling)
            video_score = torch.max(anomaly_scores).item()
            video_prediction = 1 if video_score > threshold else 0
            
            # Clip-level predictions
            clip_predictions = (anomaly_scores > threshold).int().cpu().numpy()
            clip_scores = anomaly_scores.cpu().numpy()
        
        inference_time = time.time() - start_time
        
        # Prepare results
        results = {
            'video_path': video_path,
            'video_score': video_score,
            'video_prediction': video_prediction,
            'video_label': 'FIGHT' if video_prediction == 1 else 'NO-FIGHT',
            'clip_scores': clip_scores,
            'clip_predictions': clip_predictions,
            'threshold': threshold,
            'inference_time': inference_time,
            'num_clips': len(clip_scores)
        }
        
        return results
    
    def print_results(self, results):
        """Print prediction results in a nice format"""
        print("\n" + "="*60)
        print("🥊 FIGHT DETECTION RESULTS")
        print("="*60)
        print(f"Video: {os.path.basename(results['video_path'])}")
        print(f"Prediction: {results['video_label']}")
        print(f"Confidence: {results['video_score']:.4f}")
        print(f"Threshold: {results['threshold']}")
        print(f"Inference time: {results['inference_time']:.2f}s")
        print(f"Number of clips: {results['num_clips']}")
        
        # Clip-level analysis
        fight_clips = np.sum(results['clip_predictions'])
        print(f"Fight clips: {fight_clips}/{results['num_clips']} ({fight_clips/results['num_clips']*100:.1f}%)")
        
        # Show top 5 most suspicious clips
        top_indices = np.argsort(results['clip_scores'])[-5:][::-1]
        print(f"\nTop 5 most suspicious clips:")
        for i, idx in enumerate(top_indices):
            print(f"  {i+1}. Clip {idx+1}: {results['clip_scores'][idx]:.4f}")
        
        print("="*60)
    
    def save_results(self, results, output_path):
        """Save results to file"""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        save_results = results.copy()
        save_results['clip_scores'] = save_results['clip_scores'].tolist()
        save_results['clip_predictions'] = save_results['clip_predictions'].tolist()
        
        with open(output_path, 'w') as f:
            json.dump(save_results, f, indent=2)
        
        print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fight Detection Inference")
    parser.add_argument(
        "--video_path",
        type=str,
        default="test1.mp4",
        help="Path to input video file"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./checkpoints/stage1_best.pth",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save results JSON (optional)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help="Device to use (auto-detect if not specified)"
    )
    
    args = parser.parse_args()
    
    # Check if video exists
    if not os.path.exists(args.video_path):
        print(f"❌ Video file not found: {args.video_path}")
        return
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        print(f"❌ Checkpoint not found: {args.checkpoint_path}")
        print("Available checkpoints:")
        checkpoint_dir = os.path.dirname(args.checkpoint_path)
        if os.path.exists(checkpoint_dir):
            for file in os.listdir(checkpoint_dir):
                if file.endswith('.pth'):
                    print(f"  - {os.path.join(checkpoint_dir, file)}")
        return
    
    try:
        # Initialize detector
        print("🚀 Initializing Fight Detector...")
        detector = FightDetector(
            checkpoint_path=args.checkpoint_path,
            device=args.device
        )
        
        # Run inference
        print(f"🎬 Processing video: {args.video_path}")
        results = detector.predict_video(
            video_path=args.video_path,
            threshold=args.threshold
        )
        
        # Print results
        detector.print_results(results)
        
        # Save results if requested
        if args.output_path:
            detector.save_results(results, args.output_path)
        
        print("\n✅ Inference completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
