"""
Offline Feature Extraction theo paper 2209.11477v1
Cắt video → clips T=32 → I3D → mix5c → spatial avg-pool → vector 2048/clip → save .npy
Uses existing R2Plus1DBackbone from models/
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
import argparse
from torchvision import transforms
from PIL import Image
from pathlib import Path

from models.r3d_backbone import R2Plus1DBackbone


def detect_environment():
    """Auto-detect running environment (local, Kaggle, Colab, etc.)"""
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        return 'kaggle'
    elif 'COLAB_GPU' in os.environ:
        return 'colab' 
    else:
        return 'local'


def get_default_paths(environment='local'):
    """Get default paths based on environment"""
    if environment == 'kaggle':
        # Kaggle dataset paths
        data_paths = [
            '/kaggle/input/fight-detection-dataset',
            '/kaggle/input/new-youtube',
            '/kaggle/input',  # General input path
            './input'  # Local input for Kaggle
        ]
        output_path = '/kaggle/working/features_i3d'
    elif environment == 'colab':
        # Google Colab paths
        data_paths = [
            '/content/drive/MyDrive/fight_detect/new_youtube',
            '/content/new_youtube',
            './new_youtube'
        ]
        output_path = '/content/drive/MyDrive/fight_detect/features_i3d'
    else:
        # Local development paths
        data_paths = [
            './new_youtube',
            '../new_youtube',
            '../../data/new_youtube'
        ]
        output_path = './features_i3d'
    
    return data_paths, output_path


def find_dataset_path(possible_paths):
    """Find the actual dataset path from possible locations"""
    for path in possible_paths:
        if os.path.exists(path):
            # Check if it contains expected subdirectories
            if (os.path.exists(os.path.join(path, 'train_data')) or 
                os.path.exists(os.path.join(path, 'test_data'))):
                return path
    return None


class I3DFeatureExtractor:
    """Extract I3D features từ video clips"""
    
    def __init__(self, device='auto'):
        # Auto-detect device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
                print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                self.device = 'cpu'
                print("CUDA not available, using CPU")
        else:
            self.device = device
            
        # Detect environment
        self.environment = detect_environment()
        print(f"Running environment: {self.environment}")
        
        self.model = R2Plus1DBackbone(pretrained=True).to(self.device)
        self.model.eval()
        
        # Transform cho input
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"I3D Feature Extractor initialized on {self.device}")
    
    def extract_video_features(self, video_path, clip_length=32, stride=16):
        """
        Extract features từ 1 video
        
        Args:
            video_path: Path to video file
            clip_length: T=32 frames per clip
            stride: Stride between clips
            
        Returns:
            features: (num_clips, 2048) numpy array
        """
        # Load video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return None
        
        # Read all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        if len(frames) < clip_length:
            print(f"Video too short: {video_path} ({len(frames)} frames)")
            return None
        
        # Extract clips
        clip_features = []
        
        for start_idx in range(0, len(frames) - clip_length + 1, stride):
            clip_frames = frames[start_idx:start_idx + clip_length]
            
            # Process clip
            processed_clip = self._process_clip(clip_frames)
            if processed_clip is not None:
                # Extract I3D features
                with torch.no_grad():
                    features = self.model(processed_clip)
                    # Lấy mix5c level (final_level) và spatial avg pool
                    final_features = features['final_level']  # (1, 1024, T', H', W')
                    
                    # Spatial average pooling
                    pooled_features = torch.mean(final_features, dim=[3, 4])  # (1, 1024, T')
                    # Temporal average pooling 
                    pooled_features = torch.mean(pooled_features, dim=2)  # (1, 1024)
                    
                    # Convert to 2048 dim để match paper (có thể dùng projection layer)
                    if pooled_features.shape[1] != 2048:
                        # Simple expansion to 2048
                        expanded = torch.cat([pooled_features, pooled_features], dim=1)  # (1, 2048)
                        clip_features.append(expanded.cpu().numpy())
                    else:
                        clip_features.append(pooled_features.cpu().numpy())
        
        if len(clip_features) == 0:
            return None
        
        # Stack all clip features
        video_features = np.vstack(clip_features)  # (num_clips, 2048)
        
        return video_features
    
    def _process_clip(self, frames):
        """Process một clip thành tensor"""
        try:
            processed_frames = []
            
            for frame in frames:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                # Apply transform
                frame_tensor = self.transform(frame_pil)
                processed_frames.append(frame_tensor)
            
            # Stack frames: (T, C, H, W)
            clip_tensor = torch.stack(processed_frames, dim=0)
            
            # Reshape to (1, C, T, H, W) for I3D
            clip_tensor = clip_tensor.permute(1, 0, 2, 3).unsqueeze(0)
            
            return clip_tensor.to(self.device)
        
        except Exception as e:
            print(f"Error processing clip: {e}")
            return None


def extract_dataset_features(data_root=None, output_dir=None, clip_length=32, stride=16):
    """Extract features cho toàn bộ dataset với auto-detection"""
    
    # Detect environment
    environment = detect_environment()
    print(f"Detected environment: {environment}")
    
    # Auto-detect paths if not provided
    if data_root is None:
        possible_data_paths, default_output = get_default_paths(environment)
        data_root = find_dataset_path(possible_data_paths)
        
        if data_root is None:
            print("❌ Dataset not found in any expected location!")
            print("Searched paths:")
            for path in possible_data_paths:
                print(f"  - {path}")
            print("\n💡 Please specify --data_root manually or upload dataset to expected location")
            return False
        else:
            print(f"✅ Found dataset at: {data_root}")
    
    if output_dir is None:
        _, output_dir = get_default_paths(environment)
    
    # Initialize extractor
    extractor = I3DFeatureExtractor(device='auto')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Get all video files
    video_dirs = ['train_data', 'test_data']
    all_videos = []
    
    for video_dir in video_dirs:
        video_path = os.path.join(data_root, video_dir)
        if os.path.exists(video_path):
            videos = glob.glob(os.path.join(video_path, '*.mp4'))
            all_videos.extend(videos)
            print(f"Found {len(videos)} videos in {video_dir}")
    
    if len(all_videos) == 0:
        print(f"❌ No videos found in {data_root}")
        return False
    
    print(f"📹 Total videos to process: {len(all_videos)}")
    
    # Extract features
    successful = 0
    failed = 0
    
    for video_path in tqdm(all_videos, desc="Extracting features"):
        video_name = os.path.basename(video_path)
        feature_file = os.path.join(output_dir, video_name.replace('.mp4', '.npy'))
        
        # Skip if already processed
        if os.path.exists(feature_file):
            successful += 1
            continue
        
        try:
            # Extract features
            features = extractor.extract_video_features(
                video_path, 
                clip_length=clip_length, 
                stride=stride
            )
            
            if features is not None:
                # Save features
                np.save(feature_file, features)
                successful += 1
            else:
                failed += 1
                
        except Exception as e:
            print(f"Error processing {video_name}: {e}")
            failed += 1
    
    print(f"Feature extraction completed!")
    print(f"Successful: {successful}, Failed: {failed}")
    print(f"Features saved in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Extract I3D features from videos')
    parser.add_argument('--data_root', type=str, default=None,
                       help='Path to video dataset (auto-detect if not specified)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for features (auto-detect if not specified)')
    parser.add_argument('--clip_length', type=int, default=32,
                       help='Number of frames per clip (paper default: 32)')
    parser.add_argument('--stride', type=int, default=16,
                       help='Stride between clips (paper default: 16)')
    parser.add_argument('--environment', type=str, choices=['auto', 'local', 'kaggle', 'colab'], 
                       default='auto', help='Specify environment for path detection')
    
    args = parser.parse_args()
    
    print("🚀 I3D Feature Extraction for Paper 2209.11477v1")
    print("=" * 50)
    
    # Override auto-detection if specified
    if args.environment != 'auto':
        os.environ['FORCE_ENVIRONMENT'] = args.environment
    
    try:
        success = extract_dataset_features(
            data_root=args.data_root,
            output_dir=args.output_dir,
            clip_length=args.clip_length,
            stride=args.stride
        )
        
        if success:
            print("\n🎉 Feature extraction completed successfully!")
        else:
            print("\n❌ Feature extraction failed!")
            
    except Exception as e:
        print(f"\n💥 Error during feature extraction: {e}")
        print("\n💡 Troubleshooting tips:")
        print("  1. Check if dataset path is correct")
        print("  2. Ensure sufficient disk space for output")
        print("  3. Verify CUDA/GPU availability if needed")
        print("  4. Check video file permissions")
        
    print("\n📋 Usage examples:")
    print("  Local: python extract_i3d_features.py")
    print("  Kaggle: python extract_i3d_features.py --environment kaggle")
    print("  Custom: python extract_i3d_features.py --data_root /path/to/data --output_dir /path/to/output")


def extract_all_features(input_dir=None, output_dir=None, clip_length=32, temporal_stride=16, batch_size=4, num_workers=2):
    """
    Wrapper function để compatibility với run_pipeline.py
    """
    return extract_dataset_features(
        data_root=input_dir,
        output_dir=output_dir, 
        clip_length=clip_length,
        stride=temporal_stride
    )


if __name__ == "__main__":
    main()