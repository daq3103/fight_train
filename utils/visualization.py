"""
Visualization utilities for I3D + MTN Anomaly Detection
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import os


def plot_training_curves(train_losses, val_losses, save_path=None):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(12, 5))
    
    # Loss curves
    plt.subplot(1, 2, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss difference
    plt.subplot(1, 2, 2)
    if len(train_losses) == len(val_losses):
        loss_diff = np.array(val_losses) - np.array(train_losses)
        plt.plot(epochs, loss_diff, 'g-', label='Val Loss - Train Loss', linewidth=2)
        plt.title('Overfitting Indicator')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Difference')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_anomaly_scores_timeline(frame_scores, video_labels=None, frame_labels=None, 
                                 threshold=0.5, fps=30, save_path=None):
    """Plot anomaly scores over time"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    time_axis = np.arange(len(frame_scores)) / fps
    
    # Top plot: Anomaly scores
    axes[0].plot(time_axis, frame_scores, 'b-', linewidth=2, label='Anomaly Score')
    axes[0].axhline(y=threshold, color='r', linestyle='--', alpha=0.7, label=f'Threshold ({threshold})')
    
    # Highlight anomalous regions
    anomalous_frames = frame_scores > threshold
    if np.any(anomalous_frames):
        axes[0].fill_between(time_axis, 0, 1, where=anomalous_frames, 
                            alpha=0.3, color='red', label='Predicted Anomaly')
    
    axes[0].set_ylabel('Anomaly Score')
    axes[0].set_title('Anomaly Detection Timeline')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # Bottom plot: Ground truth if available
    if frame_labels is not None:
        axes[1].plot(time_axis, frame_labels, 'g-', linewidth=2, label='Ground Truth')
        axes[1].fill_between(time_axis, 0, 1, where=frame_labels > 0.5, 
                            alpha=0.3, color='green', label='True Anomaly')
        axes[1].set_ylabel('Ground Truth')
        axes[1].set_ylim([0, 1])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].remove()
        fig.subplots_adjust(hspace=0.3)
    
    axes[-1].set_xlabel('Time (seconds)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_attention_weights(attention_weights, frame_scores, save_path=None):
    """Visualize attention weights alongside anomaly scores"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    time_steps = range(len(attention_weights))
    
    # Attention weights
    ax1.bar(time_steps, attention_weights, alpha=0.7, color='skyblue')
    ax1.set_title('MIL Attention Weights')
    ax1.set_ylabel('Attention Weight')
    ax1.grid(True, alpha=0.3)
    
    # Anomaly scores
    ax2.plot(time_steps, frame_scores, 'r-', linewidth=2, marker='o', markersize=4)
    ax2.set_title('Frame-level Anomaly Scores')
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Anomaly Score')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_anomaly_heatmap(frame_scores, video_shape, save_path=None):
    """Create heatmap visualization of anomaly scores"""
    # Reshape scores to match video temporal structure
    T = len(frame_scores)
    
    # Create a 2D heatmap (time vs score intensity)
    heatmap_data = frame_scores.reshape(-1, 1)
    
    plt.figure(figsize=(15, 4))
    sns.heatmap(heatmap_data.T, cmap='Reds', cbar_kws={'label': 'Anomaly Score'},
                xticklabels=False, yticklabels=['Anomaly\nScore'])
    plt.title('Temporal Anomaly Heatmap')
    plt.xlabel('Frame Index')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_predictions(predictions, targets, save_dir=None):
    """Comprehensive visualization of model predictions"""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Extract data
    video_scores = predictions.get('video_anomaly_score', [])
    frame_scores = predictions.get('frame_anomaly_scores', [])
    frame_predictions = predictions.get('frame_predictions', [])
    attention_weights = predictions.get('attention_weights', None)
    
    # Plot for each video in batch
    for i, (v_score, f_scores) in enumerate(zip(video_scores, frame_scores)):
        fig = plt.figure(figsize=(15, 10))
        
        # Video-level score
        plt.subplot(3, 1, 1)
        plt.bar(['Video'], [v_score], color='red' if v_score > 0.5 else 'green', alpha=0.7)
        plt.title(f'Video {i}: Overall Anomaly Score = {v_score:.3f}')
        plt.ylabel('Anomaly Score')
        plt.ylim([0, 1])
        
        # Frame-level scores
        plt.subplot(3, 1, 2)
        time_steps = range(len(f_scores))
        plt.plot(time_steps, f_scores, 'b-', linewidth=2, marker='o', markersize=3)
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Threshold')
        plt.fill_between(time_steps, 0, 1, where=np.array(f_scores) > 0.5, 
                        alpha=0.3, color='red', label='Anomalous Frames')
        plt.title('Frame-level Anomaly Scores')
        plt.xlabel('Frame Index')
        plt.ylabel('Anomaly Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])
        
        # Attention weights if available
        if attention_weights is not None and i < len(attention_weights):
            plt.subplot(3, 1, 3)
            att_weights = attention_weights[i]
            plt.bar(time_steps, att_weights, alpha=0.7, color='skyblue')
            plt.title('MIL Attention Weights')
            plt.xlabel('Frame Index')
            plt.ylabel('Attention Weight')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, f'video_{i}_predictions.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def create_video_with_annotations(video_path, frame_scores, output_path, 
                                threshold=0.5, fps=None):
    """Create annotated video with anomaly scores overlay"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    if fps is None:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx < len(frame_scores):
            score = frame_scores[frame_idx]
            is_anomaly = score > threshold
            
            # Add text overlay
            color = (0, 0, 255) if is_anomaly else (0, 255, 0)  # Red for anomaly, green for normal
            text = f"Anomaly: {score:.3f}" if is_anomaly else f"Normal: {score:.3f}"
            
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, color, 2, cv2.LINE_AA)
            
            # Add colored border
            if is_anomaly:
                cv2.rectangle(frame, (0, 0), (width-1, height-1), color, 5)
            
            # Add score bar
            bar_width = int(width * 0.3)
            bar_height = 20
            bar_x = width - bar_width - 10
            bar_y = 10
            
            # Background bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (64, 64, 64), -1)
            
            # Score bar
            score_width = int(bar_width * score)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + score_width, bar_y + bar_height), 
                         color, -1)
            
            # Threshold line
            thresh_x = bar_x + int(bar_width * threshold)
            cv2.line(frame, (thresh_x, bar_y), (thresh_x, bar_y + bar_height), 
                    (255, 255, 255), 2)
        
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()
    print(f"Annotated video saved to: {output_path}")


def plot_feature_distribution(features, labels, save_path=None):
    """Plot distribution of extracted features"""
    # Assume features is (N, D) and labels is (N,)
    if len(features.shape) > 2:
        # Flatten spatial/temporal dimensions
        features = features.reshape(features.shape[0], -1)
    
    # PCA for dimensionality reduction if needed
    if features.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
    else:
        features_2d = features
    
    plt.figure(figsize=(10, 8))
    
    # Plot points colored by label
    unique_labels = np.unique(labels)
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=colors[i % len(colors)], label=f'Class {label}', alpha=0.6)
    
    plt.title('Feature Distribution (PCA)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_confusion_matrix_plot(y_true, y_pred, class_names=None, save_path=None):
    """Create confusion matrix visualization"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # Test visualization functions
    
    # Generate sample data
    frame_scores = np.random.beta(2, 5, 100)  # More normal frames
    frame_scores[20:30] = np.random.beta(5, 2, 10)  # Anomalous region
    frame_scores[60:70] = np.random.beta(5, 2, 10)  # Another anomalous region
    
    attention_weights = np.random.dirichlet(np.ones(len(frame_scores)))
    
    # Test timeline plot
    plot_anomaly_scores_timeline(frame_scores, threshold=0.5, fps=30)
    
    # Test attention visualization
    visualize_attention_weights(attention_weights, frame_scores)
    
    # Test training curves
    train_losses = [1.0 - 0.8 * np.exp(-i/10) + 0.1 * np.random.random() for i in range(50)]
    val_losses = [1.0 - 0.7 * np.exp(-i/12) + 0.15 * np.random.random() for i in range(50)]
    plot_training_curves(train_losses, val_losses)
    
    print("Visualization tests completed!")