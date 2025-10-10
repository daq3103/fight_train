"""
Training script cho Stage 1: MIL learning với uniform grouping
Follow paper 2209.11477v1 - ranking loss + sparsity + smoothness
"""

import torch
import torch.optim as optim
import numpy as np
import os
import time
import argparse

from models.complete_model import CompleteModel, R3D_MTN_Trainer
from data.stage1_dataset import create_stage1_dataloaders
from utils.metrics import AnomalyMetrics
from config import Config


class Stage1Trainer(R3D_MTN_Trainer):
    """Stage 1 MIL Trainer using R3D_MTN_Trainer base"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create model using CompleteModel with feature_extract_mode
        model = CompleteModel(
            backbone_type="r2plus1d_18",
            num_classes=config.num_classes,
            pretrained=True,
            feature_extract_mode=True,  # For offline features
        )

        # Create optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Create scheduler
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=config.step_size, gamma=config.gamma
        )

        # Initialize parent trainer
        super().__init__(model, optimizer, self.device, scheduler)

        # Loss functions for Stage 1 MIL
        self.ranking_loss = self._ranking_loss
        self.sparsity_loss = self._sparsity_loss
        self.smoothness_loss = self._smoothness_loss

        # Metrics
        self.metrics = AnomalyMetrics()

        # Best scores tracking
        self.best_auc = 0.0
        self.best_epoch = 0

        print(f"Stage 1 Trainer initialized on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def _extract_anomaly_scores(self, outputs: torch.Tensor | dict) -> torch.Tensor:
        """Convert model outputs to (B, N) anomaly score tensor."""

        if isinstance(outputs, dict):
            if "anomaly_scores" in outputs:
                scores = outputs["anomaly_scores"]
            elif "video_score" in outputs:
                scores = outputs["video_score"]
            else:
                raise KeyError("Model output dict missing 'anomaly_scores' or 'video_score'")
        else:
            probs = torch.softmax(outputs, dim=-1)
            if probs.dim() == 3:  # (B, N, num_classes)
                scores = probs[..., 1]
            elif probs.dim() == 2:  # (B, num_classes)
                scores = probs[:, 1]
            else:
                raise ValueError(f"Unexpected output tensor shape: {probs.shape}")

        if scores.dim() == 1:  # (B,) -> (B, 1)
            scores = scores.unsqueeze(1)

        return scores

    def _ranking_loss(self, normal_scores, abnormal_scores, margin=1.0):
        """
        Ranking loss - abnormal scores phải > normal scores
        Args:
            normal_scores: (B, N) scores for normal videos
            abnormal_scores: (B, N) scores for abnormal videos
            margin: Margin for ranking
        """
        # Get max scores per video (highest anomaly score)
        normal_max = torch.max(normal_scores, dim=1)[0]  # (B,)
        abnormal_max = torch.max(abnormal_scores, dim=1)[0]  # (B,)

        # Ranking loss: max(0, margin + normal_max - abnormal_max)
        loss = torch.clamp(margin + normal_max - abnormal_max, min=0.0)
        return loss.mean()

    def _sparsity_loss(self, scores, lambda_sparsity=8e-3):
        """
        Sparsity loss - encourage sparse anomaly scores
        Args:
            scores: (B, N) anomaly scores
            lambda_sparsity: Sparsity weight
        """
        # L1 norm for sparsity
        sparsity = torch.sum(torch.abs(scores)) / scores.numel()
        return lambda_sparsity * sparsity

    def _smoothness_loss(self, scores, lambda_smooth=8e-4):
        """
        Smoothness loss - encourage temporal smoothness
        Args:
            scores: (B, N) anomaly scores
            lambda_smooth: Smoothness weight
        """
        if scores.size(1) <= 1:
            return torch.tensor(0.0, device=scores.device)

        # Temporal differences
        diff = scores[:, 1:] - scores[:, :-1]  # (B, N-1)
        smoothness = torch.sum(diff**2) / diff.numel()
        return lambda_smooth * smoothness

    def train_epoch(self, train_loader, epoch):
        """Override parent's train_epoch for Stage 1 MIL training"""
        self.model.train()

        total_loss = 0.0
        total_ranking_loss = 0.0
        total_sparsity_loss = 0.0
        total_smoothness_loss = 0.0
        num_batches = 0

        for batch_idx, batch_data in enumerate(train_loader):
            # Get data
            normal_features = batch_data["normal_features"].to(
                self.device
            )  # (B, N, feature_dim)
            abnormal_features = batch_data["abnormal_features"].to(
                self.device
            )  # (B, N, feature_dim)

            batch_size = normal_features.size(0)

            # Forward pass
            self.optimizer.zero_grad()

            # Get anomaly scores
            normal_outputs = self.model(normal_features)
            abnormal_outputs = self.model(abnormal_features)

            normal_scores = self._extract_anomaly_scores(normal_outputs)
            abnormal_scores = self._extract_anomaly_scores(abnormal_outputs)

            # Compute losses
            ranking_loss = self._ranking_loss(normal_scores, abnormal_scores)
            sparsity_loss = self._sparsity_loss(
                torch.cat([normal_scores, abnormal_scores], dim=0)
            )
            smoothness_loss = self._smoothness_loss(
                torch.cat([normal_scores, abnormal_scores], dim=0)
            )

            # Total loss
            total_loss_batch = ranking_loss + sparsity_loss + smoothness_loss

            # Backward pass
            total_loss_batch.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Accumulate losses
            total_loss += total_loss_batch.item()
            total_ranking_loss += ranking_loss.item()
            total_sparsity_loss += sparsity_loss.item()
            total_smoothness_loss += smoothness_loss.item()
            num_batches += 1

            # Print progress
            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}: "
                    f"Loss={total_loss_batch.item():.4f}, "
                    f"Ranking={ranking_loss.item():.4f}, "
                    f"Sparsity={sparsity_loss.item():.4f}, "
                    f"Smooth={smoothness_loss.item():.4f}"
                )

        # Average losses
        avg_loss = total_loss / num_batches
        avg_ranking = total_ranking_loss / num_batches
        avg_sparsity = total_sparsity_loss / num_batches
        avg_smoothness = total_smoothness_loss / num_batches

        self.train_losses.append(avg_loss)

        if self.scheduler:
            self.scheduler.step()

        return {
            "total_loss": avg_loss,
            "ranking_loss": avg_ranking,
            "sparsity_loss": avg_sparsity,
            "smoothness_loss": avg_smoothness,
        }

    def validate(self, test_loader):
        """Override parent's validate for Stage 1 evaluation"""
        self.model.eval()

        # Reset metrics
        metrics = AnomalyMetrics()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_data in test_loader:
                normal_features = batch_data["normal_features"].to(self.device)
                abnormal_features = batch_data["abnormal_features"].to(self.device)

                # Forward pass
                normal_outputs = self.model(normal_features)
                abnormal_outputs = self.model(abnormal_features)

                normal_scores = self._extract_anomaly_scores(normal_outputs)
                abnormal_scores = self._extract_anomaly_scores(abnormal_outputs)

                # Compute loss for tracking
                ranking_loss = self._ranking_loss(normal_scores, abnormal_scores)
                total_loss += ranking_loss.item()
                num_batches += 1

                # Prepare data for metrics
                batch_size = normal_features.size(0)

                # Create predictions dict
                predictions = {
                    "video_score": torch.cat(
                        [
                            torch.max(normal_scores, dim=1)[0],
                            torch.max(abnormal_scores, dim=1)[0],
                        ],
                        dim=0,
                    )
                }

                # Create targets dict
                targets = {
                    "labels": torch.cat(
                        [
                            torch.zeros(
                                batch_size, dtype=torch.long, device=self.device
                            ),  # Normal = 0
                            torch.ones(
                                batch_size, dtype=torch.long, device=self.device
                            ),  # Abnormal = 1
                        ],
                        dim=0,
                    )
                }

                # Update metrics
                metrics.update(predictions, targets)

        # Compute all metrics
        all_metrics = metrics.compute_all_metrics(threshold=0.5)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.val_losses.append(avg_loss)

        return {
            "loss": avg_loss,
            "auc": all_metrics.get("video_auc_roc", 0.0),
            "accuracy": all_metrics.get("video_accuracy", 0.0),
            "precision": all_metrics.get("video_precision", 0.0),
            "recall": all_metrics.get("video_recall", 0.0),
            "f1": all_metrics.get("video_f1", 0.0),
            "threshold": 0.5,
        }

    def save_checkpoint(self, epoch, metrics, checkpoint_dir):
        """Save Stage 1 checkpoint with custom metrics"""
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Use parent's checkpoint format but add Stage 1 specific data
        filepath = os.path.join(checkpoint_dir, "stage1_latest.pth")
        super().save_checkpoint(
            filepath, epoch, best_loss=metrics.get("loss", float("inf"))
        )

        # Save best checkpoint based on AUC
        if metrics["auc"] > self.best_auc:
            self.best_auc = metrics["auc"]
            self.best_epoch = epoch
            best_path = os.path.join(checkpoint_dir, "stage1_best.pth")
            super().save_checkpoint(best_path, epoch, best_loss=metrics["auc"])
            print(f"New best AUC: {self.best_auc:.4f} at epoch {epoch}")

    def load_checkpoint(self, checkpoint_path):
        """Load Stage 1 checkpoint"""
        if os.path.exists(checkpoint_path):
            epoch, best_loss = super().load_checkpoint(checkpoint_path)
            print(f"Loaded Stage 1 checkpoint from epoch {epoch}")
            return epoch, {"loss": best_loss}
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
            return 0, None

    def train(
        self, train_loader, test_loader, num_epochs, checkpoint_dir="./checkpoints"
    ):
        """Main training loop"""
        print(f"Starting Stage 1 training for {num_epochs} epochs...")
        print(f"Checkpoint directory: {checkpoint_dir}")

        start_epoch = 0

        # Try to load latest checkpoint
        latest_checkpoint = os.path.join(checkpoint_dir, "stage1_latest.pth")
        if os.path.exists(latest_checkpoint):
            start_epoch, _ = self.load_checkpoint(latest_checkpoint)
            start_epoch += 1  # Start from next epoch

        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Evaluate
            eval_metrics = self.validate(test_loader)

            # Step scheduler
            self.scheduler.step()

            epoch_time = time.time() - epoch_start_time

            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs-1} completed in {epoch_time:.2f}s")
            print(
                f"Train Loss: {train_metrics['total_loss']:.4f} "
                f"(Ranking: {train_metrics['ranking_loss']:.4f}, "
                f"Sparsity: {train_metrics['sparsity_loss']:.4f}, "
                f"Smooth: {train_metrics['smoothness_loss']:.4f})"
            )
            print(
                f"Val Loss: {eval_metrics['loss']:.4f}, "
                f"AUC: {eval_metrics['auc']:.4f}, "
                f"Acc: {eval_metrics['accuracy']:.4f}"
            )
            print(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Save checkpoint
            self.save_checkpoint(epoch, eval_metrics, checkpoint_dir)

            print("-" * 80)

        print(f"\nTraining completed!")
        print(f"Best AUC: {self.best_auc:.4f} at epoch {self.best_epoch}")


def main():
    parser = argparse.ArgumentParser(description="Stage 1 MIL Training")
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
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
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

    args = parser.parse_args()

    # Create config
    config = Config()
    config.learning_rate = args.learning_rate
    config.batch_size = args.batch_size

    print(f"Stage 1 Training Configuration:")
    print(f"Feature directory: {args.feature_dir}")
    print(f"Data root: {args.data_root}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")

    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, test_loader = create_stage1_dataloaders(
        feature_dir=args.feature_dir,
        data_root=args.data_root,
        batch_size=args.batch_size,
        feature_dim=config.feature_dim,
        num_workers=args.num_workers,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create trainer
    trainer = Stage1Trainer(config)

    # Start training
    trainer.train(
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=args.num_epochs,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
