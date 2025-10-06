"""
I3D + MTN Anomaly Detection for Fight Detection
Complete implementation with Feature Extractor (I3D) + Anomaly Score Generator (MTN/MLP)

Usage:
    python train.py --config fight --data_path /path/to/data --epochs 50
    python train.py --config debug  # Quick test
    python evaluate.py --model_path checkpoints/best_model.pth --test_path /path/to/test
"""

# Model imports
from models.i3d_backbone import I3DBackbone
from models.mtn_anomaly import MultipleTemporalNetwork, AnomalyScoreGenerator
from models.complete_model import I3D_MTN_AnomalyDetector, I3D_MTN_Trainer

# Data imports
from data.dataset import AnomalyVideoDataset, FrameLevelAnomalyDataset

# Utility imports
from utils.metrics import AnomalyMetrics
from utils.visualization import (
    plot_training_curves,
    plot_anomaly_scores_timeline,
    visualize_predictions,
    create_video_with_annotations
)

# Configuration
from config import get_config, create_experiment_config

__version__ = "1.0.0"
__author__ = "Fight Detection Team"

# Main components for easy import
__all__ = [
    # Models
    'I3DBackbone',
    'MultipleTemporalNetwork', 
    'AnomalyScoreGenerator',
    'I3D_MTN_AnomalyDetector',
    'I3D_MTN_Trainer',
    
    # Data
    'AnomalyVideoDataset',
    'FrameLevelAnomalyDataset',
    
    # Utils
    'AnomalyMetrics',
    
    # Visualization
    'plot_training_curves',
    'plot_anomaly_scores_timeline',
    'visualize_predictions',
    'create_video_with_annotations',
    
    # Config
    'get_config',
    'create_experiment_config'
]

def quick_test():
    """Quick test function to verify installation"""
    import torch
    
    print("Testing I3D + MTN Anomaly Detection Framework...")
    
    # Test basic imports
    try:
        config = get_config("debug")
        print("✓ Config loaded successfully")
        
        # Test model creation
        model = I3D_MTN_AnomalyDetector(
            clip_length=config.CLIP_LENGTH,
            i3d_backbone=config.I3D_BACKBONE,
            mtn_hidden_dim=config.MTN_HIDDEN_DIM,
            mtn_num_scales=config.MTN_NUM_SCALES,
            mil_attention_dim=config.MIL_ATTENTION_DIM
        )
        print("✓ Model created successfully")
        
        # Test forward pass
        batch_size = 1
        clip_length = config.CLIP_LENGTH
        H, W = config.FRAME_SIZE
        
        dummy_input = torch.randn(batch_size, 3, clip_length, H, W)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print("✓ Forward pass successful")
        print(f"  Video anomaly score shape: {output['video_anomaly_score'].shape}")
        print(f"  Frame anomaly scores shape: {output['frame_anomaly_scores'].shape}")
        
        # Test trainer
        trainer = I3D_MTN_Trainer(model, config)
        print("✓ Trainer created successfully")
        
        print("\n🎉 All tests passed! Framework is ready to use.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    quick_test()