"""
Dataset cho Stage 2: Variable-length training với pseudo-labels
Trả [M, 2048] (M biến) + nhãn video để áp pseudo-labels
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from typing import List, Tuple


class Stage2Dataset(Dataset):
    """Dataset cho Stage 2 training với variable-length sequences"""

    def __init__(
        self,
        feature_dir: str,
        data_root: str,
        mode: str = "train",
        feature_dim: int = 2048,
        min_clips: int = 8,
        max_clips: int = 256,
    ):
        """
        Args:
            feature_dir: Directory chứa .npy feature files
            data_root: Root directory của new_youtube dataset
            mode: 'train' hoặc 'test'
            feature_dim: Feature dimension (2048)
            min_clips: Minimum clips per video
            max_clips: Maximum clips per video (để limit memory)
        """
        self.feature_dir = feature_dir
        self.data_root = data_root
        self.mode = mode
        self.feature_dim = feature_dim
        self.min_clips = min_clips
        self.max_clips = max_clips

        # Load video data
        self.video_data = self._load_video_data()

        print(f"Stage2 Dataset ({mode}): {len(self.video_data)} videos")
        fight_count = sum(1 for item in self.video_data if item["label"] == 1)
        print(f"  Fight: {fight_count}, No-fight: {len(self.video_data) - fight_count}")

        # Statistics
        clip_counts = [item["num_clips"] for item in self.video_data]
        print(
            f"  Clips per video: min={min(clip_counts)}, max={max(clip_counts)}, mean={np.mean(clip_counts):.1f}"
        )

    def _load_video_data(self) -> List[dict]:
        """Load video data với clip counts"""
        video_data = []

        # Get video files dựa trên mode
        if self.mode == "train":
            video_pattern = os.path.join(self.data_root, "train_data", "*.mp4")
        elif self.mode == "test":
            video_pattern = os.path.join(self.data_root, "test_data", "*.mp4")
        else:
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

            try:
                # Load features để check shape
                features = np.load(feature_file)
                num_clips, feature_dim = features.shape

                # Filter by clip count
                if num_clips < self.min_clips:
                    continue

                if num_clips > self.max_clips:
                    print(
                        f"Truncating {video_name}: {num_clips} -> {self.max_clips} clips"
                    )
                    num_clips = self.max_clips

                # Determine label từ filename
                if video_name.startswith("f_"):
                    label = 1  # Fight
                elif video_name.startswith("nof_"):
                    label = 0  # No-fight
                else:
                    continue

                video_data.append(
                    {
                        "video_name": video_name,
                        "feature_file": feature_file,
                        "label": label,
                        "num_clips": num_clips,
                    }
                )

            except Exception as e:
                print(f"Error checking {video_name}: {str(e)}")
                continue

        return video_data

    def __len__(self):
        return len(self.video_data)

    def __getitem__(self, idx):
        """
        Returns:
            features: tensor [M, feature_dim] - variable length
            label: int (0 hoặc 1) - video level label
            num_clips: int - actual number of clips
        """
        video_info = self.video_data[idx]

        try:
            # Load features
            features = np.load(video_info["feature_file"])  # (M, feature_dim)

            # Truncate if necessary
            if features.shape[0] > self.max_clips:
                features = features[: self.max_clips]

            # Convert to tensor
            features_tensor = torch.from_numpy(features).float()
            label_tensor = torch.tensor(video_info["label"], dtype=torch.long)

            return {
                "features": features_tensor,  # [M, feature_dim] - variable length
                "label": label_tensor,  # scalar - video level
                "num_clips": features.shape[0],
            }

        except Exception as e:
            print(f"Error loading {video_info['video_name']}: {str(e)}")

            # Return dummy data
            dummy_features = torch.zeros(self.min_clips, self.feature_dim)
            dummy_label = torch.tensor(0, dtype=torch.long)

            return {
                "features": dummy_features,
                "label": dummy_label,
                "num_clips": self.min_clips,
            }


def stage2_collate_fn(batch):
    """
    Collate function cho variable length sequences
    """
    # Sort batch by sequence length (longest first)
    batch = sorted(batch, key=lambda x: x["num_clips"], reverse=True)

    features_list = [item["features"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    num_clips = [item["num_clips"] for item in batch]

    # Pad sequences
    max_clips = max(num_clips)
    batch_size = len(batch)
    feature_dim = features_list[0].shape[1]

    # Create padded tensor
    padded_features = torch.zeros(batch_size, max_clips, feature_dim)
    attention_mask = torch.zeros(batch_size, max_clips, dtype=torch.bool)

    for i, (features, clips) in enumerate(zip(features_list, num_clips)):
        padded_features[i, :clips] = features
        attention_mask[i, :clips] = True

    return {
        "features": padded_features,  # (B, max_clips, feature_dim)
        "attention_mask": attention_mask,  # (B, max_clips) - True for valid clips
        "labels": labels,  # (B,) - video level labels
        "num_clips": num_clips,  # List[int] - actual clip counts
        "features_list": features_list,  # List of variable length tensors
    }


def create_stage2_dataloaders(
    feature_dir: str,
    data_root: str,
    batch_size: int = 4,
    feature_dim: int = 2048,
    min_clips: int = 8,
    max_clips: int = 256,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create Stage 2 dataloaders

    Returns:
        train_loader, test_loader
    """
    # Create datasets
    train_dataset = Stage2Dataset(
        feature_dir=feature_dir,
        data_root=data_root,
        mode="train",
        feature_dim=feature_dim,
        min_clips=min_clips,
        max_clips=max_clips,
    )

    test_dataset = Stage2Dataset(
        feature_dir=feature_dir,
        data_root=data_root,
        mode="test",
        feature_dim=feature_dim,
        min_clips=min_clips,
        max_clips=max_clips,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=stage2_collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=stage2_collate_fn,
        pin_memory=True,
    )

    return train_loader, test_loader


class PseudoLabelGenerator:
    """Generate pseudo labels cho clips dựa trên Stage 1 model predictions"""

    def __init__(self, stage1_model, device="cuda", strategy="threshold"):
        """
        Args:
            stage1_model: Trained Stage 1 model
            device: Device for computation
            strategy: 'threshold', 'top_k', hoặc 'soft'
        """
        self.stage1_model = stage1_model
        self.device = device
        self.strategy = strategy

        self.stage1_model.eval()

    def generate_pseudo_labels(self, features, video_label, threshold=0.5, top_k=None):
        """
        Generate pseudo labels cho clips

        Args:
            features: (M, feature_dim) - clip features
            video_label: Video-level label (0 hoặc 1)
            threshold: Threshold cho binary classification
            top_k: Number of top clips để label positive (for top_k strategy)

        Returns:
            pseudo_labels: (M,) - pseudo labels cho clips
        """
        with torch.no_grad():
            # Get clip-level scores từ Stage 1 model
            features_tensor = torch.from_numpy(features).float().to(self.device)

            # Expand to batch format if needed
            if features_tensor.dim() == 2:
                features_tensor = features_tensor.unsqueeze(0)  # (1, M, feature_dim)

            # Forward through Stage 1 model để get clip scores
            # Assume model returns clip-level scores
            model_outputs = self.stage1_model(features_tensor)

            clip_scores_tensor = None

            if isinstance(model_outputs, dict):
                # Prefer explicit logits if available
                clip_logits = None
                for key in ["clip_logits", "logits"]:
                    if key in model_outputs and model_outputs[key] is not None:
                        clip_logits = model_outputs[key]
                        break

                if clip_logits is not None:
                    clip_scores_tensor = torch.softmax(clip_logits, dim=-1)[..., 1]
                elif model_outputs.get("clip_scores") is not None:
                    clip_scores_tensor = model_outputs["clip_scores"].squeeze(0)
                elif model_outputs.get("anomaly_scores") is not None:
                    clip_scores_tensor = model_outputs["anomaly_scores"].squeeze(0)
            else:
                clip_logits = model_outputs
                clip_scores_tensor = torch.softmax(clip_logits, dim=-1)[..., 1]

            if clip_scores_tensor is None:
                # Fallback: uniform pseudo labels
                return self._uniform_pseudo_labels(features.shape[0], video_label)

            # Ensure 1D tensor of fight probabilities (M,)
            clip_scores_tensor = clip_scores_tensor.squeeze()
            if clip_scores_tensor.dim() == 0:
                clip_scores_tensor = clip_scores_tensor.unsqueeze(0)

            clip_scores = clip_scores_tensor.detach().cpu().numpy()

            if self.strategy == "threshold":
                pseudo_labels = (clip_scores > threshold).astype(int)
            elif self.strategy == "top_k":
                if top_k is None:
                    top_k = max(1, int(len(clip_scores) * 0.3))  # Top 30%

                pseudo_labels = np.zeros(len(clip_scores), dtype=int)
                top_indices = np.argsort(clip_scores)[-top_k:]
                pseudo_labels[top_indices] = 1
            elif self.strategy == "soft":
                # Keep probabilities for soft labels
                pseudo_labels = clip_scores
            else:
                pseudo_labels = self._uniform_pseudo_labels(
                    len(clip_scores), video_label
                )

            return pseudo_labels

    def _uniform_pseudo_labels(self, num_clips, video_label):
        """Fallback uniform pseudo labels"""
        if video_label == 1:  # Fight video
            # 60-80% clips are fight
            fight_ratio = np.random.uniform(0.6, 0.8)
            num_fight_clips = int(num_clips * fight_ratio)

            pseudo_labels = np.zeros(num_clips, dtype=int)
            fight_indices = np.random.choice(num_clips, num_fight_clips, replace=False)
            pseudo_labels[fight_indices] = 1

            return pseudo_labels
        else:  # No-fight video
            return np.zeros(num_clips, dtype=int)


if __name__ == "__main__":
    # Test Stage 2 dataset
    feature_dir = "./features_i3d"
    data_root = "./new_youtube"

    if os.path.exists(feature_dir):
        print("Testing Stage 2 Dataset...")

        # Create dataset
        dataset = Stage2Dataset(
            feature_dir=feature_dir,
            data_root=data_root,
            mode="train",
            feature_dim=2048,
            min_clips=8,
            max_clips=128,
        )

        if len(dataset) > 0:
            # Test single sample
            sample = dataset[0]
            print(f"Features shape: {sample['features'].shape}")
            print(f"Label: {sample['label']}")
            print(f"Num clips: {sample['num_clips']}")

            # Test dataloader
            train_loader, test_loader = create_stage2_dataloaders(
                feature_dir=feature_dir,
                data_root=data_root,
                batch_size=2,
                max_clips=128,
                num_workers=0,
            )

            print(f"Train batches: {len(train_loader)}")
            print(f"Test batches: {len(test_loader)}")

            # Test batch
            for batch in train_loader:
                print(
                    f"Batch features: {batch['features'].shape}"
                )  # (B, max_clips, feature_dim)
                print(
                    f"Batch attention_mask: {batch['attention_mask'].shape}"
                )  # (B, max_clips)
                print(f"Batch labels: {batch['labels'].shape}")  # (B,)
                print(f"Batch num_clips: {batch['num_clips']}")  # List[int]
                break
        else:
            print("No data found in dataset")
    else:
        print(f"Feature directory not found: {feature_dir}")
        print("Please run extract_i3d_features.py first")
