"""
Training script cho Stage 2: Pseudo-label training với variable-length sequences
Follow paper 2209.11477v1 - cross-entropy loss với frozen encoder
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import argparse
from datetime import datetime

from models.complete_model import CompleteModel, R3D_MTN_Trainer
from data.stage2_dataset import create_stage2_dataloaders, PseudoLabelGenerator
from utils.metrics import AnomalyMetrics, calculate_metrics
from config import Config


class Stage2Trainer(R3D_MTN_Trainer):
    """Stage 2 Trainer using R3D_MTN_Trainer base với pseudo-label learning"""

    def __init__(self, config, stage1_checkpoint_path):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create model
        model = CompleteModel(
            backbone_type="r2plus1d_18",
            num_classes=config.num_classes,
            pretrained=True,
            feature_extract_mode=True,  # For offline features
        )

        # Load Stage 1 model cho pseudo-label generation
        self.stage1_model = self._load_stage1_model(stage1_checkpoint_path)

        # Freeze encoder if specified
        if config.freeze_encoder_stage2:
            self._freeze_encoder(model)

        # Optimizer (only unfrozen parameters)
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(
            trainable_params,
            lr=config.learning_rate_stage2,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=config.step_size, gamma=config.gamma
        )

        # Initialize parent trainer
        super().__init__(model, optimizer, self.device, scheduler)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Pseudo label generator
        self.pseudo_label_generator = PseudoLabelGenerator(
            stage1_model=self.stage1_model,
            device=self.device,
            strategy=config.pseudo_label_strategy,
        )

        # Metrics
        self.metrics = AnomalyMetrics()

        # Best scores tracking
        self.best_auc = 0.0
        self.best_epoch = 0

        # Count trainable parameters
        trainable_params_count = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total_params_count = sum(p.numel() for p in self.model.parameters())

        print(f"Stage 2 Trainer initialized on {self.device}")
        print(f"Total parameters: {total_params_count:,}")
        print(f"Trainable parameters: {trainable_params_count:,}")
        print(f"Frozen parameters: {total_params_count - trainable_params_count:,}")

    def _load_stage1_model(self, checkpoint_path):
        """Load Stage 1 model cho pseudo-label generation"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Stage 1 checkpoint not found: {checkpoint_path}")

        # Create Stage 1 model
        stage1_model = CompleteModel(
            backbone_type="r2plus1d_18",
            num_classes=self.config.num_classes,
            pretrained=True,
            feature_extract_mode=True,
        ).to(self.device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        stage1_model.load_state_dict(checkpoint["model_state_dict"])
        stage1_model.eval()

        print(f"Loaded Stage 1 model from: {checkpoint_path}")
        print(f"Stage 1 epoch: {checkpoint['epoch']}")
        print(f"Stage 1 metrics: {checkpoint.get('metrics', 'N/A')}")

        return stage1_model

    def _freeze_encoder(self, model):
        """Freeze encoder layers"""
        frozen_layers = []

        # Freeze feature encoder (MTN layers)
        for name, param in model.named_parameters():
            if "encoder" in name or "attention" in name or "temporal" in name:
                param.requires_grad = False
                frozen_layers.append(name)

        print(f"Frozen {len(frozen_layers)} encoder layers:")
        for layer in frozen_layers:
            print(f"  - {layer}")

    def _generate_clip_pseudo_labels(self, features_batch, video_labels_batch):
        """
        Generate pseudo labels cho clips trong batch

        Args:
            features_batch: List of (M_i, feature_dim) tensors
            video_labels_batch: (B,) video labels

        Returns:
            pseudo_labels_batch: List of (M_i,) pseudo labels
        """
        pseudo_labels_batch = []

        for features, video_label in zip(features_batch, video_labels_batch):
            # Convert to numpy
            features_np = features.cpu().numpy()
            video_label_np = video_label.cpu().item()

            # Generate pseudo labels
            pseudo_labels = self.pseudo_label_generator.generate_pseudo_labels(
                features=features_np,
                video_label=video_label_np,
                threshold=self.config.pseudo_label_threshold,
                top_k=self.config.pseudo_label_top_k,
            )

            # Convert back to tensor
            pseudo_labels_tensor = (
                torch.from_numpy(pseudo_labels).long().to(self.device)
            )
            pseudo_labels_batch.append(pseudo_labels_tensor)

        return pseudo_labels_batch

    def train_epoch(self, train_loader, epoch):
        """Train một epoch"""
        self.model.train()

        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        for batch_idx, batch_data in enumerate(train_loader):
            features_list = batch_data[
                "features_list"
            ]  # List of variable length tensors
            video_labels = batch_data["labels"]  # (B,) video labels

            batch_size = len(features_list)

            # Generate pseudo labels cho clips
            pseudo_labels_batch = self._generate_clip_pseudo_labels(
                features_list, video_labels
            )

            # Forward pass và loss computation
            self.optimizer.zero_grad()

            batch_loss = 0.0
            batch_accuracy = 0.0

            for features, pseudo_labels in zip(features_list, pseudo_labels_batch):
                features = features.to(self.device)  # (M, feature_dim)
                pseudo_labels = pseudo_labels.to(self.device)  # (M,)

                # Expand features for batch processing
                features_batch = features.unsqueeze(0)  # (1, M, feature_dim)

                # Forward pass
                outputs = self.model(features_batch)

                if isinstance(outputs, dict):
                    clip_logits = outputs.get("clip_logits", outputs.get("logits"))
                else:
                    clip_logits = outputs

                if clip_logits is None:
                    # Fallback: create dummy logits
                    clip_logits = torch.randn(
                        1, features.size(0), 2, device=self.device
                    )

                clip_logits = clip_logits.squeeze(0)  # (M, num_classes)

                # Cross-entropy loss
                loss = self.criterion(clip_logits, pseudo_labels)
                batch_loss += loss

                # Accuracy
                predictions = torch.argmax(clip_logits, dim=1)
                accuracy = (predictions == pseudo_labels).float().mean()
                batch_accuracy += accuracy

            # Average across videos trong batch
            batch_loss = batch_loss / batch_size
            batch_accuracy = batch_accuracy / batch_size

            # Backward pass
            batch_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Accumulate metrics
            total_loss += batch_loss.item()
            total_accuracy += batch_accuracy.item()
            num_batches += 1

            # Print progress
            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}: "
                    f"Loss={batch_loss.item():.4f}, "
                    f"Acc={batch_accuracy.item():.4f}"
                )

        # Average metrics
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches

        return {"loss": avg_loss, "accuracy": avg_accuracy}

    def evaluate(self, test_loader):
        """Evaluate model trên test set"""
        self.model.eval()

        all_video_scores = []
        all_video_labels = []
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_data in test_loader:
                features_list = batch_data["features_list"]
                video_labels = batch_data["labels"]

                batch_size = len(features_list)

                # Generate pseudo labels (for loss computation)
                pseudo_labels_batch = self._generate_clip_pseudo_labels(
                    features_list, video_labels
                )

                batch_loss = 0.0

                for i, (features, pseudo_labels) in enumerate(
                    zip(features_list, pseudo_labels_batch)
                ):
                    features = features.to(self.device)
                    pseudo_labels = pseudo_labels.to(self.device)

                    # Forward pass
                    features_batch = features.unsqueeze(0)
                    outputs = self.model(features_batch)

                    if isinstance(outputs, dict):
                        clip_logits = outputs.get("clip_logits", outputs.get("logits"))
                        anomaly_scores = outputs.get("anomaly_scores")
                    else:
                        clip_logits = outputs
                        anomaly_scores = None

                    if clip_logits is not None:
                        clip_logits = clip_logits.squeeze(0)
                        loss = self.criterion(clip_logits, pseudo_labels)
                        batch_loss += loss

                        # Get video-level score (max probability of fight class)
                        clip_probs = torch.softmax(clip_logits, dim=1)  # (M, 2)
                        fight_probs = clip_probs[:, 1]  # (M,) probabilities of fight
                        video_score = torch.max(fight_probs).item()
                    else:
                        # Fallback to anomaly scores
                        if anomaly_scores is not None:
                            video_score = torch.max(anomaly_scores.squeeze(0)).item()
                        else:
                            video_score = 0.5  # Random guess

                    all_video_scores.append(video_score)
                    all_video_labels.append(video_labels[i].item())

                total_loss += (batch_loss / batch_size).item()
                num_batches += 1

        # Calculate metrics using AnomalyMetrics
        metrics = AnomalyMetrics()

        # Create predictions and targets for metrics
        predictions = {
            "video_score": torch.tensor(all_video_scores, device=self.device)
        }
        targets = {"labels": torch.tensor(all_video_labels, device=self.device)}

        # Update metrics
        metrics.update(predictions, targets)

        # Compute all metrics
        all_metrics = metrics.compute_all_metrics(threshold=0.5)

        avg_loss = total_loss / num_batches

        return {
            "loss": avg_loss,
            "auc": all_metrics.get("video_auc_roc", 0.0),
            "accuracy": all_metrics.get("video_accuracy", 0.0),
            "precision": all_metrics.get("video_precision", 0.0),
            "recall": all_metrics.get("video_recall", 0.0),
            "f1": all_metrics.get("video_f1", 0.0),
        }

    def save_checkpoint(self, epoch, metrics, checkpoint_dir):
        """Save model checkpoint"""
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config.__dict__,
        }

        # Save latest checkpoint
        latest_path = os.path.join(checkpoint_dir, "stage2_latest.pth")
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if metrics["auc"] > self.best_auc:
            self.best_auc = metrics["auc"]
            self.best_epoch = epoch
            best_path = os.path.join(checkpoint_dir, "stage2_best.pth")
            torch.save(checkpoint, best_path)
            print(f"New best AUC: {self.best_auc:.4f} at epoch {epoch}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            epoch = checkpoint["epoch"]
            metrics = checkpoint["metrics"]

            print(f"Loaded Stage 2 checkpoint from epoch {epoch}")
            print(f"Previous metrics: {metrics}")

            return epoch, metrics
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
            return 0, None

    def train(
        self, train_loader, test_loader, num_epochs, checkpoint_dir="./checkpoints"
    ):
        """Main training loop"""
        print(f"Starting Stage 2 training for {num_epochs} epochs...")
        print(f"Checkpoint directory: {checkpoint_dir}")

        start_epoch = 0

        # Try to load latest checkpoint
        latest_checkpoint = os.path.join(checkpoint_dir, "stage2_latest.pth")
        if os.path.exists(latest_checkpoint):
            start_epoch, _ = self.load_checkpoint(latest_checkpoint)
            start_epoch += 1

        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Evaluate
            eval_metrics = self.evaluate(test_loader)

            # Step scheduler
            self.scheduler.step()

            epoch_time = time.time() - epoch_start_time

            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs-1} completed in {epoch_time:.2f}s")
            print(
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}"
            )
            print(
                f"Val Loss: {eval_metrics['loss']:.4f}, "
                f"Val AUC: {eval_metrics['auc']:.4f}, "
                f"Val Acc: {eval_metrics['accuracy']:.4f}"
            )
            print(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Save checkpoint
            self.save_checkpoint(epoch, eval_metrics, checkpoint_dir)

            print("-" * 80)

        print(f"\nStage 2 training completed!")
        print(f"Best AUC: {self.best_auc:.4f} at epoch {self.best_epoch}")


def main():
    parser = argparse.ArgumentParser(description="Stage 2 Pseudo-label Training")
    parser.add_argument(
        "--feature_dir",
        type=str,
        default="./features_i3d",
        help="Directory containing I3D features",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./new_youtube",
        help="Root directory of dataset",
    )
    parser.add_argument(
        "--stage1_checkpoint",
        type=str,
        required=True,
        help="Path to Stage 1 best checkpoint",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--num_workers", type=int, default=2, help="Number of data loader workers"
    )
    parser.add_argument(
        "--max_clips", type=int, default=128, help="Maximum clips per video"
    )

    args = parser.parse_args()

    # Validate Stage 1 checkpoint
    if not os.path.exists(args.stage1_checkpoint):
        print(f"Error: Stage 1 checkpoint not found: {args.stage1_checkpoint}")
        print("Please train Stage 1 first or provide correct path")
        return

    # Create config
    config = Config()
    config.learning_rate_stage2 = args.learning_rate
    config.batch_size = args.batch_size

    print(f"Stage 2 Training Configuration:")
    print(f"Feature directory: {args.feature_dir}")
    print(f"Data root: {args.data_root}")
    print(f"Stage 1 checkpoint: {args.stage1_checkpoint}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max clips per video: {args.max_clips}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")

    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, test_loader = create_stage2_dataloaders(
        feature_dir=args.feature_dir,
        data_root=args.data_root,
        batch_size=args.batch_size,
        feature_dim=config.feature_dim,
        max_clips=args.max_clips,
        num_workers=args.num_workers,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create trainer
    trainer = Stage2Trainer(config, args.stage1_checkpoint)

    # Start training
    trainer.train(
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=args.num_epochs,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
