"""
Metrics for Anomaly Detection Evaluation
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_curve
import matplotlib.pyplot as plt


class AnomalyMetrics:
    """Comprehensive metrics for anomaly detection evaluation"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all stored predictions and targets"""
        self.video_scores = []
        self.video_labels = []
        self.frame_scores = []
        self.frame_labels = []
        self.classification_preds = []
        self.classification_labels = []

    def update(self, predictions, targets):
        """
        Update metrics with new batch of predictions and targets

        Args:
            predictions: Dictionary from model containing:
                - video_score: (B,) video-level anomaly scores
                - anomaly_scores: (B, T) frame-level anomaly scores
                - classification_logits: (B, num_classes) classification outputs
            targets: Dictionary containing:
                - labels: (B,) video-level labels
                - frame_labels: (B, T) frame-level labels (optional)
        """
        # Video-level metrics
        if "video_score" in predictions:
            video_scores = predictions["video_score"].detach().cpu().numpy()
            video_labels = targets["labels"].detach().cpu().numpy()

            self.video_scores.extend(video_scores)
            self.video_labels.extend(video_labels)

        # Frame-level metrics
        if "anomaly_scores" in predictions and "frame_labels" in targets:
            frame_scores = predictions["anomaly_scores"].detach().cpu().numpy()
            frame_labels = targets["frame_labels"].detach().cpu().numpy()

            # Flatten for frame-level evaluation
            self.frame_scores.extend(frame_scores.flatten())
            self.frame_labels.extend(frame_labels.flatten())

        # Classification metrics
        if "classification_logits" in predictions:
            cls_preds = (
                torch.argmax(predictions["classification_logits"], dim=1)
                .detach()
                .cpu()
                .numpy()
            )
            cls_labels = targets["labels"].detach().cpu().numpy()

            self.classification_preds.extend(cls_preds)
            self.classification_labels.extend(cls_labels)

    def compute_video_metrics(self, threshold=0.5):
        """Compute video-level anomaly detection metrics"""
        if not self.video_scores:
            return {}

        scores = np.array(self.video_scores)
        labels = np.array(self.video_labels)

        # AUC metrics
        try:
            auc_roc = roc_auc_score(labels, scores)
            auc_pr = average_precision_score(labels, scores)
        except ValueError:
            auc_roc = 0.0
            auc_pr = 0.0

        # Threshold-based metrics
        predictions = (scores > threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        return {
            "video_auc_roc": auc_roc,
            "video_auc_pr": auc_pr,
            "video_precision": precision,
            "video_recall": recall,
            "video_f1": f1,
            "video_specificity": specificity,
            "video_accuracy": accuracy,
            "video_tp": tp,
            "video_fp": fp,
            "video_tn": tn,
            "video_fn": fn,
        }

    def compute_frame_metrics(self, threshold=0.5):
        """Compute frame-level anomaly detection metrics"""
        if not self.frame_scores:
            return {}

        scores = np.array(self.frame_scores)
        labels = np.array(self.frame_labels)

        # Remove invalid scores/labels
        valid_mask = ~(np.isnan(scores) | np.isnan(labels))
        scores = scores[valid_mask]
        labels = labels[valid_mask]

        if len(scores) == 0:
            return {}

        # AUC metrics
        try:
            auc_roc = roc_auc_score(labels, scores)
            auc_pr = average_precision_score(labels, scores)
        except ValueError:
            auc_roc = 0.0
            auc_pr = 0.0

        # Threshold-based metrics
        predictions = (scores > threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        return {
            "frame_auc_roc": auc_roc,
            "frame_auc_pr": auc_pr,
            "frame_precision": precision,
            "frame_recall": recall,
            "frame_f1": f1,
            "frame_specificity": specificity,
            "frame_accuracy": accuracy,
            "frame_tp": tp,
            "frame_fp": fp,
            "frame_tn": tn,
            "frame_fn": fn,
        }

    def compute_classification_metrics(self):
        """Compute classification accuracy metrics"""
        if not self.classification_preds:
            return {}

        preds = np.array(self.classification_preds)
        labels = np.array(self.classification_labels)

        accuracy = np.mean(preds == labels)

        # Per-class accuracy
        unique_labels = np.unique(labels)
        class_accuracies = {}
        for label in unique_labels:
            mask = labels == label
            if np.sum(mask) > 0:
                class_acc = np.mean(preds[mask] == labels[mask])
                class_accuracies[f"class_{label}_accuracy"] = class_acc

        return {"classification_accuracy": accuracy, **class_accuracies}

    def compute_all_metrics(self, threshold=0.5):
        """Compute all available metrics"""
        metrics = {}

        # Video-level metrics
        video_metrics = self.compute_video_metrics(threshold)
        metrics.update(video_metrics)

        # Frame-level metrics
        frame_metrics = self.compute_frame_metrics(threshold)
        metrics.update(frame_metrics)

        # Classification metrics
        cls_metrics = self.compute_classification_metrics()
        metrics.update(cls_metrics)

        return metrics

    def find_best_threshold(self, metric="f1", level="video"):
        """Find best threshold based on specified metric"""
        if level == "video":
            scores = np.array(self.video_scores)
            labels = np.array(self.video_labels)
        else:
            scores = np.array(self.frame_scores)
            labels = np.array(self.frame_labels)

        if len(scores) == 0:
            return 0.5, 0.0

        # Test multiple thresholds
        thresholds = np.linspace(0.0, 1.0, 101)
        best_threshold = 0.5
        best_score = 0.0

        for threshold in thresholds:
            predictions = (scores > threshold).astype(int)

            try:
                tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

                if metric == "f1":
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    score = (
                        2 * (precision * recall) / (precision + recall)
                        if (precision + recall) > 0
                        else 0.0
                    )
                elif metric == "accuracy":
                    score = (tp + tn) / (tp + tn + fp + fn)
                elif metric == "precision":
                    score = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                elif metric == "recall":
                    score = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                else:
                    continue

                if score > best_score:
                    best_score = score
                    best_threshold = threshold

            except ValueError:
                continue

        return best_threshold, best_score

    def plot_roc_curve(self, save_path=None, level="video"):
        """Plot ROC curve"""
        if level == "video":
            scores = np.array(self.video_scores)
            labels = np.array(self.video_labels)
            title = "Video-level ROC Curve"
        else:
            scores = np.array(self.frame_scores)
            labels = np.array(self.frame_labels)
            title = "Frame-level ROC Curve"

        if len(scores) == 0:
            return

        try:
            fpr, tpr, _ = roc_curve(labels, scores)
            auc = roc_auc_score(labels, scores)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {auc:.3f})")
            plt.plot([0, 1], [0, 1], "k--", linewidth=1)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(title)
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close()
            else:
                plt.show()

        except ValueError as e:
            print(f"Error plotting ROC curve: {e}")

    def plot_pr_curve(self, save_path=None, level="video"):
        """Plot Precision-Recall curve"""
        if level == "video":
            scores = np.array(self.video_scores)
            labels = np.array(self.video_labels)
            title = "Video-level Precision-Recall Curve"
        else:
            scores = np.array(self.frame_scores)
            labels = np.array(self.frame_labels)
            title = "Frame-level Precision-Recall Curve"

        if len(scores) == 0:
            return

        try:
            precision, recall, _ = precision_recall_curve(labels, scores)
            auc_pr = average_precision_score(labels, scores)

            plt.figure(figsize=(8, 6))
            plt.plot(
                recall, precision, linewidth=2, label=f"PR Curve (AUC = {auc_pr:.3f})"
            )
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(title)
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close()
            else:
                plt.show()

        except ValueError as e:
            print(f"Error plotting PR curve: {e}")


def evaluate_model(model, data_loader, device, threshold=0.5):
    """Evaluate model on a dataset"""
    model.eval()
    metrics = AnomalyMetrics()

    with torch.no_grad():
        for videos, targets in data_loader:
            videos = videos.to(device)

            # Move targets to device
            device_targets = {}
            for key, value in targets.items():
                if isinstance(value, torch.Tensor):
                    device_targets[key] = value.to(device)
                else:
                    device_targets[key] = value

            # Forward pass
            predictions = model(videos)

            # Update metrics
            metrics.update(predictions, device_targets)

    # Compute all metrics
    all_metrics = metrics.compute_all_metrics(threshold)

    return all_metrics, metrics


def calculate_metrics(labels, predictions):
    """Simple metrics calculation for binary classification"""
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    labels = np.array(labels)
    predictions = np.array(predictions)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average="binary", zero_division=0)
    recall = recall_score(labels, predictions, average="binary", zero_division=0)
    f1 = f1_score(labels, predictions, average="binary", zero_division=0)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
