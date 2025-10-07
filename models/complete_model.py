"""
Complete I3D + MTN Anomaly Detection Model
Combining Feature Extractor (I3D) + Anomaly Score Generator (MTN/MLP)
"""

import sys
import os

sys.path.append(os.path.dirname(__file__))  # Thêm models directory vào path

import torch
import torch.nn as nn
import torch.nn.functional as F
from r3d_backbone import R2Plus1DBackbone
from mtn_anomaly import (
    MultipleTemporalNetwork,
    AnomalyScoreGenerator,
    AnomalyDetectionLoss,
)


class R3D_MTN_AnomalyDetector(nn.Module):
    """Complete anomaly detection model with R3D feature extraction and MTN analysis"""

    def __init__(
        self,
        num_classes=2,
        r3d_pretrained=True,
        mtn_hidden_dim=512,
        asg_hidden_dims=[256, 128],
        use_mil=True,
        dropout_prob=0.3,
        freeze_r3d_layers=0,
    ):
        super(R3D_MTN_AnomalyDetector, self).__init__()

        self.num_classes = num_classes  # khởi tạo số lớp phát hiện
        self.use_mil = use_mil  # sử dụng Multiple Instance Learning

        # 1. R3D Feature Extractor
        self.r3d_backbone = R2Plus1DBackbone(pretrained=r3d_pretrained)

        # Freeze early layers if specified
        if freeze_r3d_layers > 0:
            self._freeze_r3d_layers(freeze_r3d_layers)

        # Define feature channels from R(2+1)D layers
        self.feature_channels = {
            "low_level": 192,
            "mid_level": 480,
            "high_level": 832,
            "final_level": 1024,
        }

        # dùng MTN để phân tích temporal các features trích xuất được từ R3D
        self.mtn = MultipleTemporalNetwork(
            feature_channels_dict=self.feature_channels, hidden_dim=mtn_hidden_dim
        )

        # 3. tính điểm bất thường và phân loại
        self.anomaly_generator = AnomalyScoreGenerator(
            input_dim=mtn_hidden_dim,
            hidden_dims=asg_hidden_dims,
            num_classes=num_classes,
            use_mil=use_mil,
            dropout_prob=dropout_prob,
        )

        # 4. Loss function
        self.criterion = AnomalyDetectionLoss(
            anomaly_weight=1.0, classification_weight=1.0, smoothness_weight=0.1
        )

    def _freeze_r3d_layers(self, num_layers):
        """Freeze early layers of R3D for fine-tuning"""
        layers_to_freeze = [
            "Conv3d_1a_7x7",
            "Conv3d_2b_1x1",
            "Conv3d_2c_3x3",
            "Mixed_3b",
            "Mixed_3c",
        ][:num_layers]

        for name, module in self.r3d_backbone.named_modules():
            for freeze_name in layers_to_freeze:
                if freeze_name in name:
                    for param in module.parameters():
                        param.requires_grad = False

    def forward(self, x, return_features=False):

        # 1. Extract features with R3D
        r3d_features = self.r3d_backbone(x)

        # 2. Temporal analysis with MTN
        temporal_features = self.mtn(r3d_features)

        # 3. Anomaly scoring and classification
        results = self.anomaly_generator(
            temporal_features, return_scores=True, return_classification=True
        )

        if return_features:
            results["r3d_features"] = r3d_features
            results["temporal_features"] = temporal_features

        return results

    def compute_loss(self, predictions, targets):
        """Compute combined loss"""
        return self.criterion(predictions, targets)

    def predict_anomaly(self, x, threshold=0.5):  # ngưỡng để phân loại bất thường

        self.eval()
        with torch.no_grad():
            results = self.forward(x)

            # Video-level prediction
            video_scores = results["video_score"]
            video_predictions = (video_scores > threshold).int()

            # Frame-level predictions
            frame_scores = results["anomaly_scores"]
            frame_predictions = (frame_scores > threshold).int()

            # Classification predictions
            if "classification_logits" in results:
                cls_probs = F.softmax(results["classification_logits"], dim=1)
                cls_predictions = torch.argmax(cls_probs, dim=1)
            else:
                cls_probs = None
                cls_predictions = None

            return {
                "video_anomaly_score": video_scores.cpu().numpy(),
                "video_prediction": video_predictions.cpu().numpy(),
                "frame_anomaly_scores": frame_scores.cpu().numpy(),
                "frame_predictions": frame_predictions.cpu().numpy(),
                "classification_probs": (
                    cls_probs.cpu().numpy() if cls_probs is not None else None
                ),
                "classification_prediction": (
                    cls_predictions.cpu().numpy()
                    if cls_predictions is not None
                    else None
                ),
                "attention_weights": (
                    results["attention_weights"].cpu().numpy()
                    if results["attention_weights"] is not None
                    else None
                ),
            }


class R3D_MTN_Trainer:
    """Trainer class for R3D + MTN anomaly detection model"""

    def __init__(self, model, optimizer, device="cuda", scheduler=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_loss_dict = {}

        for batch_idx, (videos, targets) in enumerate(train_loader):
            videos = videos.to(self.device)

            # Prepare targets
            if isinstance(targets, dict):
                targets = {k: v.to(self.device) for k, v in targets.items()}
            else:
                targets = {"labels": targets.to(self.device)}

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(videos)

            # Compute loss
            loss, loss_dict = self.model.compute_loss(predictions, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Accumulate loss components
            for key, value in loss_dict.items():
                if key not in total_loss_dict:
                    total_loss_dict[key] = 0.0
                total_loss_dict[key] += value

            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        # Average losses
        avg_loss = total_loss / len(train_loader)
        avg_loss_dict = {k: v / len(train_loader) for k, v in total_loss_dict.items()}

        self.train_losses.append(avg_loss)

        if self.scheduler:
            self.scheduler.step()

        return avg_loss, avg_loss_dict

    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        total_loss_dict = {}

        with torch.no_grad():
            for videos, targets in val_loader:
                videos = videos.to(self.device)

                if isinstance(targets, dict):
                    targets = {k: v.to(self.device) for k, v in targets.items()}
                else:
                    targets = {"labels": targets.to(self.device)}

                predictions = self.model(videos)
                loss, loss_dict = self.model.compute_loss(predictions, targets)

                total_loss += loss.item()

                for key, value in loss_dict.items():
                    if key not in total_loss_dict:
                        total_loss_dict[key] = 0.0
                    total_loss_dict[key] += value

        avg_loss = total_loss / len(val_loader)
        avg_loss_dict = {k: v / len(val_loader) for k, v in total_loss_dict.items()}

        self.val_losses.append(avg_loss)

        return avg_loss, avg_loss_dict

    def save_checkpoint(self, filepath, epoch, best_loss=None):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_loss": best_loss,
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])

        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint.get("epoch", 0), checkpoint.get("best_loss", float("inf"))


if __name__ == "__main__":
    # Test complete model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = R3D_MTN_AnomalyDetector(
        num_classes=2,
        r3d_pretrained=True,  # Set to True when pretrained weights available
        mtn_hidden_dim=512,
        asg_hidden_dims=[256, 128],
        use_mil=True,
        dropout_prob=0.3,
    )

    # Test input
    x = torch.randn(2, 3, 32, 224, 224).to(
        device
    )  # (batch, channels, time, height, width)
    model = model.to(device)

    # Forward pass
    results = model(x, return_features=True)

    print("I3D + MTN Anomaly Detection Results:")
    for key, value in results.items():
        if hasattr(value, "shape"):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, dict):
            print(f"  {key}: dict with keys {list(value.keys())}")
        else:
            print(f"  {key}: {type(value)}")


class CompleteModel(nn.Module):
    """Simplified model for feature-based training"""

    def __init__(
        self,
        backbone_type="r2plus1d_18",
        num_classes=2,
        pretrained=True,
        feature_extract_mode=True,
    ):
        super(CompleteModel, self).__init__()

        self.backbone_type = backbone_type
        self.num_classes = num_classes
        self.feature_extract_mode = feature_extract_mode

        if feature_extract_mode:
            # For feature-based training, we only need a classifier
            # Assuming input features are 2048-dimensional (from I3D)
            self.classifier = nn.Linear(2048, num_classes)
        else:
            # Full model with backbone (for future use)
            self.backbone = R2Plus1DBackbone(pretrained=pretrained)
            self.classifier = nn.Linear(1024, num_classes)  # R3D final features

    def forward(self, x):
        if self.feature_extract_mode:
            # x is already extracted features (B, feature_dim) or (B, T, feature_dim)
            if x.dim() == 3:  # (B, T, feature_dim)
                # Process each clip
                B, T, D = x.shape
                x = x.view(-1, D)  # (B*T, feature_dim)
                logits = self.classifier(x)  # (B*T, num_classes)
                logits = logits.view(B, T, -1)  # (B, T, num_classes)
            else:  # (B, feature_dim)
                logits = self.classifier(x)  # (B, num_classes)
        else:
            # Full forward pass through backbone
            features = self.backbone(x)
            logits = self.classifier(features["final_level"])

        return logits
