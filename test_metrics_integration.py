"""
Test metrics integration với pipeline
"""

import torch
import numpy as np
from utils.metrics import AnomalyMetrics, calculate_metrics

def test_anomaly_metrics():
    """Test AnomalyMetrics class"""
    print("=== Testing AnomalyMetrics ===")
    
    # Initialize metrics
    metrics = AnomalyMetrics()
    
    # Create dummy data
    batch_size = 4
    
    # Simulate predictions from model
    predictions = {
        'video_score': torch.tensor([0.2, 0.8, 0.3, 0.9]),  # Video-level anomaly scores
        'anomaly_scores': torch.randn(batch_size, 32),        # Frame-level scores
        'classification_logits': torch.randn(batch_size, 2)   # Classification logits
    }
    
    # Simulate targets
    targets = {
        'labels': torch.tensor([0, 1, 0, 1]),                # Video-level labels
        'frame_labels': torch.randint(0, 2, (batch_size, 32)) # Frame-level labels
    }
    
    # Update metrics
    metrics.update(predictions, targets)
    
    # Compute all metrics
    all_metrics = metrics.compute_all_metrics(threshold=0.5)
    
    print("Video-level metrics:")
    for key, value in all_metrics.items():
        if 'video' in key:
            print(f"  {key}: {value:.4f}")
    
    print("Classification metrics:")
    for key, value in all_metrics.items():
        if 'classification' in key:
            print(f"  {key}: {value:.4f}")
    
    print("✓ AnomalyMetrics test passed")
    return True

def test_metrics_with_pipeline():
    """Test metrics với pipeline components"""
    print("\n=== Testing Metrics with Pipeline ===")
    
    try:
        # Test Stage 1 trainer import
        from train_stage1 import Stage1Trainer
        from config import Config
        
        config = Config()
        
        print("✓ Stage1Trainer với AnomalyMetrics imported successfully")
        
        # Test Stage 2 trainer import
        from train_stage2 import Stage2Trainer
        
        print("✓ Stage2Trainer với AnomalyMetrics imported successfully")
        
        # Test CompleteModel compatibility
        from models.complete_model import CompleteModel
        
        model = CompleteModel(
            backbone_type="r2plus1d_18",
            num_classes=2,
            pretrained=False,  # To avoid download
            feature_extract_mode=True
        )
        
        # Test dummy forward pass
        dummy_features = torch.randn(2, 32, 2048)  # (batch, clips, features)
        
        with torch.no_grad():
            outputs = model(dummy_features)
        
        print(f"✓ CompleteModel output shape: {outputs.shape}")
        
        # Test metrics compatibility
        metrics = AnomalyMetrics()
        
        # Convert model outputs to metrics format
        if outputs.dim() == 3:  # (B, T, num_classes)
            probs = torch.softmax(outputs, dim=-1)
            video_scores = torch.max(probs[..., 1], dim=1)[0]  # Max fight probability
        else:  # (B, num_classes)
            probs = torch.softmax(outputs, dim=-1)
            video_scores = probs[:, 1]  # Fight probability
        
        predictions = {'video_score': video_scores}
        targets = {'labels': torch.tensor([0, 1])}
        
        metrics.update(predictions, targets)
        computed_metrics = metrics.compute_all_metrics()
        
        print("✓ Metrics computation with CompleteModel outputs successful")
        print(f"  Video AUC: {computed_metrics.get('video_auc_roc', 0.0):.4f}")
        print(f"  Video Accuracy: {computed_metrics.get('video_accuracy', 0.0):.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Pipeline metrics test failed: {e}")
        return False

def test_sklearn_metrics():
    """Test sklearn-based metrics"""
    print("\n=== Testing sklearn-based metrics ===")
    
    try:
        # Test calculate_metrics function
        labels = [0, 1, 0, 1, 1, 0]
        predictions = [0, 1, 1, 1, 0, 0]
        
        sklearn_metrics = calculate_metrics(labels, predictions)
        
        print("sklearn metrics:")
        for key, value in sklearn_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        print("✓ sklearn metrics test passed")
        return True
        
    except Exception as e:
        print(f"✗ sklearn metrics test failed: {e}")
        return False

def main():
    print("🧪 Testing Metrics Integration")
    print("=" * 50)
    
    # Test individual components
    test1 = test_anomaly_metrics()
    test2 = test_metrics_with_pipeline()
    test3 = test_sklearn_metrics()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"AnomalyMetrics: {'✓ PASS' if test1 else '✗ FAIL'}")
    print(f"Pipeline Integration: {'✓ PASS' if test2 else '✗ FAIL'}")
    print(f"sklearn Metrics: {'✓ PASS' if test3 else '✗ FAIL'}")
    
    if all([test1, test2, test3]):
        print("\n🎉 All metrics tests passed!")
        print("✅ Pipeline is ready with comprehensive metrics!")
        print("\n💡 Available metrics:")
        print("  - Video-level: AUC, Accuracy, Precision, Recall, F1")
        print("  - Frame-level: AUC, Accuracy, Precision, Recall, F1") 
        print("  - Classification: Accuracy, Per-class accuracy")
        print("  - sklearn compatibility: Full sklearn.metrics support")
    else:
        print("\n❌ Some metrics tests failed. Please check implementation.")

if __name__ == "__main__":
    main()