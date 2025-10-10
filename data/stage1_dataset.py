"""
Dataset cho Stage 1: MIL training với 32 clips/video
Uniform grouping + averaging để chuẩn hoá về N=32 clip-features
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from typing import List, Tuple


class Stage1Dataset(Dataset):
    """Dataset cho Stage 1 MIL training"""

    def __init__(
        self,
        feature_dir: str,
        data_root: str,
        mode: str = "train",
        target_clips: int = 32,
        feature_dim: int = 1024,
    ):
        """
        Args:
            feature_dir: Directory chứa .npy feature files
            data_root: Root directory của new_youtube dataset
            mode: 'train' hoặc 'test'
            target_clips: Số clips chuẩn hoá (N=32)
            feature_dim: Dimension của features (1024)
        """
        self.feature_dir = feature_dir
        self.data_root = data_root
        self.mode = mode
        self.target_clips = target_clips
        self.feature_dim = feature_dim

        # Load video data
        self.video_data = self._load_video_data()

        print(f"Stage1 Dataset ({mode}): {len(self.video_data)} videos")
        fight_count = sum(1 for item in self.video_data if item["label"] == 1)
        print(f"  Fight: {fight_count}, No-fight: {len(self.video_data) - fight_count}")

        self.normal_videos = [v for v in self.video_data if v["label"] == 0]
        self.abnormal_videos = [v for v in self.video_data if v["label"] == 1]
        self.num_pairs = min(len(self.normal_videos), len(self.abnormal_videos))

    def _load_video_data(self) -> List[dict]:
        """Load video file paths và labels"""
        video_data = []

        # Get video files dựa trên mode
        if self.mode == "train":
            video_pattern = os.path.join(self.data_root, "train_data", "*.mp4")
        elif self.mode == "test":
            video_pattern = os.path.join(self.data_root, "test_data", "*.mp4")
        else:
            # All videos
            train_pattern = os.path.join(self.data_root, "train_data", "*.mp4")
            test_pattern = os.path.join(self.data_root, "test_data", "*.mp4")
            video_files = glob.glob(train_pattern) + glob.glob(test_pattern)

        if self.mode in ["train", "test"]:
            video_files = glob.glob(video_pattern)

        for video_path in video_files:
            video_name = os.path.basename(video_path)
            feature_file = os.path.join(
                self.feature_dir, video_name.replace(".mp4", ".npy")
            )

            # Check if feature file exists
            if not os.path.exists(feature_file):
                continue

            # Determine label từ filename
            if video_name.startswith("f_"):
                label = 1  # Fight
            elif video_name.startswith("nof_"):
                label = 0  # No-fight
            else:
                continue

            video_data.append(
                {"video_name": video_name, "feature_file": feature_file, "label": label}
            )

        return video_data

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        normal_info = self.normal_videos[idx % len(self.normal_videos)]
        abnormal_info = self.abnormal_videos[idx % len(self.abnormal_videos)]

        normal_features = torch.from_numpy(
            np.load(normal_info["feature_file"])
        ).float()
        abnormal_features = torch.from_numpy(
            np.load(abnormal_info["feature_file"])
        ).float()

        return {
            "normal_features": normal_features,
            "abnormal_features": abnormal_features,
            "normal_name": normal_info["video_name"],
            "abnormal_name": abnormal_info["video_name"],
        }


def create_stage1_dataloaders(
    feature_dir: str,
    data_root: str,
    batch_size: int = 8,
    target_clips: int = 32,
    feature_dim: int = 1024,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create Stage 1 dataloaders

    Returns:
        train_loader, test_loader
    """
    # Create datasets
    train_dataset = Stage1Dataset(
        feature_dir=feature_dir,
        data_root=data_root,
        mode="train",
        target_clips=target_clips,
        feature_dim=feature_dim,
    )

    test_dataset = Stage1Dataset(
        feature_dir=feature_dir,
        data_root=data_root,
        mode="test",
        target_clips=target_clips,
        feature_dim=feature_dim,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


if __name__ == "__main__":
    feature_dir = "./features_i3d"
    data_root = "./new_youtube"

    if os.path.exists(feature_dir):
        print("Testing Stage 1 Dataset...")

        dataset = Stage1Dataset(
            feature_dir=feature_dir,
            data_root=data_root,
            mode="train",
            target_clips=32,
            feature_dim=1024,
        )

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample keys: {list(sample.keys())}")
            print(f"Normal shape: {sample['normal_features'].shape}")
            print(f"Abnormal shape: {sample['abnormal_features'].shape}")

            train_loader, test_loader = create_stage1_dataloaders(
                feature_dir=feature_dir,
                data_root=data_root,
                batch_size=4,
                num_workers=0,
            )

            print(f"Train batches: {len(train_loader)}")
            print(f"Test batches: {len(test_loader)}")

            batch = next(iter(train_loader))
            print(f"Batch normal: {batch['normal_features'].shape}")
            print(f"Batch abnormal: {batch['abnormal_features'].shape}")
        else:
            print("No data found in dataset")
    else:
        print(f"Feature directory not found: {feature_dir}")
