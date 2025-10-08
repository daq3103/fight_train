import torch
import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
import argparse
from torchvision import transforms
from PIL import Image

from models.r3d_backbone import R2Plus1DBackbone


class I3DFeatureExtractor:
    """Extract I3D features từ video clips"""

    def __init__(self):

        if torch.cuda.is_available():
            self.device = "cuda"
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            print("CUDA not available, using CPU")

        self.model = R2Plus1DBackbone(pretrained=True).to(self.device)
        self.model.eval()

        # Transform cho input
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        print(f"I3D Feature Extractor initialized on {self.device}")

    def extract_video_features(self, video_path, target_clips=32):
        """Paper-compliant feature extraction"""
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        total_frames = len(frames)
        if total_frames < 1:
            return None
        
        clip_features = []
        
        for i in range(target_clips):
            # Calculate segment boundaries
            start_ratio = i / target_clips
            end_ratio = (i + 1) / target_clips
            
            start_idx = int(start_ratio * total_frames)
            end_idx = int(end_ratio * total_frames)
            
            # Handle edge case
            if start_idx >= total_frames:
                start_idx = total_frames - 1
            if end_idx > total_frames:
                end_idx = total_frames
            if start_idx == end_idx:
                end_idx = start_idx + 1
                
            # Extract segment frames
            segment_frames = frames[start_idx:end_idx]
            
            # Prepare 32 frames for I3D (paper uses T=32)
            if len(segment_frames) < 32:
                # Repeat frames in segment
                repeated_frames = []
                for j in range(32):
                    frame_idx = j % len(segment_frames)
                    repeated_frames.append(segment_frames[frame_idx])
                final_frames = repeated_frames
            else:
                # Sample 32 frames from segment
                indices = np.linspace(0, len(segment_frames)-1, 32).astype(int)
                final_frames = [segment_frames[idx] for idx in indices]
            
            # Process and extract features
            processed_clip = self._process_clip(final_frames)
            if processed_clip is not None:
                with torch.no_grad():
                    features = self.model(processed_clip)
                    final_features = features["final_level"]
                    pooled_features = torch.mean(final_features, dim=[3, 4])
                    pooled_features = torch.mean(pooled_features, dim=2)
                    clip_features.append(pooled_features.cpu().numpy())
        
        if len(clip_features) != target_clips:
            print(f"Warning: Expected {target_clips} clips, got {len(clip_features)}")
            return None
        
        # đảm bảo đúng shape (32, feature_dim) 
        video_features = np.vstack(clip_features)  # (32, 1024)
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


def extract_dataset_features(
    data_root=None, output_dir=None, target_clips=32
):
    """Extract features cho toàn bộ dataset với uniform segmentation"""

    # Auto-detect paths if not provided
    if data_root is None:
        data_root = "./new_youtube"
        if not os.path.exists(data_root):
            print("không tìm thấy vị trí dataset!")
            return False
        else:
            print(f"Found dataset at: {data_root}")

    if output_dir is None:
        output_dir = "./features_i3d"

    # Initialize extractor
    extractor = I3DFeatureExtractor()

    # tạo output dir nếu chưa có
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # lấy danh sách folder train_data, test_data
    video_dirs = ["train_data", "test_data"]
    all_videos = []

    for video_dir in video_dirs:
        video_path = os.path.join(data_root, video_dir)
        if os.path.exists(video_path):
            videos = glob.glob(os.path.join(video_path, "*.mp4"))
            all_videos.extend(videos)
            print(f"Found {len(videos)} videos in {video_dir}")

    if len(all_videos) == 0:
        print(f"❌ không tìm thấy video nào trong {data_root}")
        return False

    print(f"📹 Tổng số video cần xử lý: {len(all_videos)}")

    # Extract features
    successful = 0
    failed = 0

    for video_path in tqdm(all_videos, desc="Extracting features"):
        video_name = os.path.basename(video_path)
        feature_file = os.path.join(output_dir, video_name.replace(".mp4", ".npy"))

        if os.path.exists(feature_file):
            successful += 1
            continue

        try:
            # ✅ FIXED: Use uniform segmentation
            features = extractor.extract_video_features(
                video_path, target_clips=target_clips
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
    return successful > 0  # ✅ Return success status


def main():
    parser = argparse.ArgumentParser(description="Extract I3D features from videos")
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to video dataset (auto-detect if not specified)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for features (auto-detect if not specified)",    
    )
    parser.add_argument(
        "--target_clips",  # ✅ Changed from clip_length
        type=int,
        default=32,
        help="Number of clips per video (paper default: 32)",
    )

    args = parser.parse_args()

    print("🚀 I3D Feature Extraction for Paper 2209.11477v1")
    print("=" * 50)

    try:
        success = extract_dataset_features(
            data_root=args.data_root,
            output_dir=args.output_dir,
            target_clips=args.target_clips,  # ✅ Fixed parameter
        )

        if success:
            print("\n🎉 Feature extraction completed successfully!")
        else:
            print("\n❌ Feature extraction failed!")

    except Exception as e:
        print(f"\n💥 Error during feature extraction: {e}")


if __name__ == "__main__":
    main()
