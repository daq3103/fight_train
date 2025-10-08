"""
I3D + MTN Anomaly Detection for Fight Detection
Complete implementation with Feature Extractor (I3D) + Anomaly Score Generator (MTN/MLP)

Usage:
    python train.py --config fight --data_path /path/to/data --epochs 50
    python train.py --config debug  # Quick test
    python evaluate.py --model_path checkpoints/best_model.pth --test_path /path/to/test
"""

__version__ = "1.0.0"
__author__ = "Fight Detection Team"

__all__ = [
]
