"""
Script để kiểm tra các checkpoint có sẵn
"""

import os
import torch
import glob


def check_checkpoints(checkpoint_dir="./checkpoints"):
    """Kiểm tra các checkpoint có sẵn"""
    
    print("🔍 Checking available checkpoints...")
    print(f"Directory: {checkpoint_dir}")
    
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return []
    
    # Tìm tất cả file .pth
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    
    if not checkpoint_files:
        print("❌ No checkpoint files found")
        return []
    
    print(f"✅ Found {len(checkpoint_files)} checkpoint(s):")
    
    checkpoints_info = []
    
    for checkpoint_path in checkpoint_files:
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract info
            epoch = checkpoint.get('epoch', 'unknown')
            best_loss = checkpoint.get('best_loss', 'unknown')
            
            # Check if it's a model state dict
            has_model = 'model_state_dict' in checkpoint
            
            info = {
                'path': checkpoint_path,
                'filename': os.path.basename(checkpoint_path),
                'epoch': epoch,
                'best_loss': best_loss,
                'has_model': has_model,
                'size_mb': os.path.getsize(checkpoint_path) / (1024*1024)
            }
            
            checkpoints_info.append(info)
            
            print(f"\n📁 {info['filename']}")
            print(f"   Size: {info['size_mb']:.1f} MB")
            print(f"   Epoch: {epoch}")
            print(f"   Best Loss: {best_loss}")
            print(f"   Has Model: {has_model}")
            
        except Exception as e:
            print(f"❌ Error loading {checkpoint_path}: {e}")
    
    return checkpoints_info


def recommend_checkpoint(checkpoints_info):
    """Đề xuất checkpoint tốt nhất"""
    
    if not checkpoints_info:
        print("❌ No valid checkpoints found")
        return None
    
    # Ưu tiên stage1_best.pth
    for info in checkpoints_info:
        if 'best' in info['filename'].lower():
            print(f"\n🎯 Recommended checkpoint: {info['filename']}")
            return info['path']
    
    # Nếu không có best, chọn latest
    for info in checkpoints_info:
        if 'latest' in info['filename'].lower():
            print(f"\n🎯 Recommended checkpoint: {info['filename']}")
            return info['path']
    
    # Nếu không có, chọn file đầu tiên
    print(f"\n🎯 Recommended checkpoint: {checkpoints_info[0]['filename']}")
    return checkpoints_info[0]['path']


if __name__ == "__main__":
    # Check checkpoints
    checkpoints = check_checkpoints()
    
    if checkpoints:
        # Recommend best checkpoint
        recommended = recommend_checkpoint(checkpoints)
        
        if recommended:
            print(f"\n💡 To use this checkpoint for inference:")
            print(f"   python infer.py --checkpoint_path {recommended}")
            print(f"   python quick_infer.py  # (will use {recommended})")
    else:
        print("\n💡 To train a model first:")
        print("   python train_stage1.py")
