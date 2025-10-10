"""
Quick inference script - đơn giản hơn để test nhanh
"""

import torch
import numpy as np
import os
from models.complete_model import CompleteModel
from extract_i3d_features import I3DFeatureExtractor
from config import Config


def quick_predict(video_path="test.mp4", checkpoint_path="./checkpoints/stage1_best.pth"):
    """Quick prediction function"""
    
    print(f"🎬 Processing: {video_path}")
    
    # Check files
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    config = Config()
    model = CompleteModel(
        backbone_type="r2plus1d_18",
        num_classes=2,
        pretrained=True,
        feature_extract_mode=True,
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("✅ Model loaded successfully")
    
    # Extract features
    extractor = I3DFeatureExtractor()
    features = extractor.extract_video_features(video_path, target_clips=32)
    
    if features is None:
        print("❌ Failed to extract features")
        return
    
    print(f"✅ Features extracted: {features.shape}")
    
    # Predict
    features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(features_tensor)
        probs = torch.softmax(logits, dim=-1)
        anomaly_scores = probs[0, :, 1]  # Fight probabilities
        
        video_score = torch.max(anomaly_scores).item()
        video_prediction = "FIGHT" if video_score > 0.5 else "NO-FIGHT"
    
    # Results
    print("\n" + "="*50)
    print("🥊 QUICK PREDICTION RESULTS")
    print("="*50)
    print(f"Video: {os.path.basename(video_path)}")
    print(f"Prediction: {video_prediction}")
    print(f"Confidence: {video_score:.4f}")
    print(f"Threshold: 0.5")
    
    fight_clips = torch.sum(anomaly_scores > 0.5).item()
    print(f"Fight clips: {fight_clips}/32 ({fight_clips/32*100:.1f}%)")
    
    # Top 3 most suspicious clips
    top_scores, top_indices = torch.topk(anomaly_scores, 3)
    print(f"\nTop 3 suspicious clips:")
    for i, (score, idx) in enumerate(zip(top_scores, top_indices)):
        print(f"  {i+1}. Clip {idx.item()+1}: {score.item():.4f}")
    
    print("="*50)
    
    return {
        'prediction': video_prediction,
        'confidence': video_score,
        'fight_clips': fight_clips,
        'total_clips': 32
    }


if __name__ == "__main__":
    # Quick test
    result = quick_predict()
    
    if result:
        print(f"\n🎉 Final result: {result['prediction']} (confidence: {result['confidence']:.4f})")
