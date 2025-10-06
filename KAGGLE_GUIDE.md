# 🥊 Fight Detection on Kaggle - Quick Start Guide

## 🚀 **1-Click Setup for Kaggle**

### **Step 1: Upload Code to Kaggle**
1. Zip toàn bộ project folder
2. Upload lên Kaggle Datasets
3. Create new Kaggle Notebook
4. Add your uploaded code dataset

### **Step 2: Auto Setup**
```python
# Run this in first cell
import sys
sys.path.append('/kaggle/working')
exec(open('kaggle_setup.py').read())
```

### **Step 3: Extract Features**
```bash
!python extract_i3d_features.py --environment kaggle
```

### **Step 4: Train Models**
```bash
# Stage 1 MIL Training
!python train_stage1.py --feature_dir /kaggle/working/features_i3d --batch_size 8 --num_epochs 50

# Stage 2 Pseudo-label Training  
!python train_stage2.py --stage1_checkpoint /kaggle/working/checkpoints/stage1/stage1_best.pth --batch_size 4 --num_epochs 30
```

### **Step 5: Or Run Full Pipeline**
```bash
!python run_pipeline.py --data_root /kaggle/input/fight-detection-dataset --feature_dir /kaggle/working/features_i3d
```

---

## 📂 **Expected Kaggle Structure**

```
/kaggle/
├── input/
│   └── fight-detection-dataset/    # Your uploaded dataset
│       ├── train_data/
│       │   ├── f_*.mp4           # Fight videos
│       │   └── nof_*.mp4         # Non-fight videos
│       └── test_data/
│           ├── f_*.mp4
│           └── nof_*.mp4
├── working/                        # Output directory
│   ├── features_i3d/              # Extracted features
│   ├── checkpoints/                # Trained models
│   └── kaggle_config.py           # Auto-generated config
```

---

## 🔧 **Auto-Detection Features**

✅ **Environment Detection**: Kaggle vs Colab vs Local  
✅ **Path Auto-Configuration**: No manual path setup needed  
✅ **GPU Auto-Detection**: Automatically uses available GPU  
✅ **Dataset Auto-Discovery**: Finds dataset in any input location  
✅ **Error Handling**: Clear error messages with troubleshooting tips  

---

## 📊 **Expected Performance on Kaggle**

| Component | Time | Memory | Notes |
|-----------|------|--------|-------|
| Feature Extraction | ~30-60 min | ~8GB RAM | GPU recommended |
| Stage 1 Training | ~20-40 min | ~6GB GPU | 50 epochs |
| Stage 2 Training | ~15-30 min | ~4GB GPU | 30 epochs |
| **Total Pipeline** | **~1-2 hours** | **~8GB GPU** | Full training |

---

## 🎯 **Kaggle-Specific Optimizations**

### **Reduced Resource Usage**
```python
# Optimized for Kaggle GPU limits
config.batch_size = 8          # Reduced from 16
config.num_epochs = 50         # Reduced from 100  
config.num_workers = 2         # Kaggle-safe
config.max_clips = 128         # Memory efficient
```

### **Checkpointing Strategy**
- Auto-save every epoch
- Resume from checkpoint if interrupted
- Best model preservation

### **Memory Management**
- Gradient accumulation for large batches
- Dynamic batch sizing
- Efficient feature loading

---

## 🚨 **Troubleshooting**

### **Dataset Not Found**
```bash
# Check available datasets
!ls /kaggle/input/

# Manual path specification
!python extract_i3d_features.py --data_root /kaggle/input/your-dataset-name
```

### **GPU Memory Issues**
```bash
# Reduce batch size
!python train_stage1.py --batch_size 4 --num_epochs 30
```

### **Time Limits**
```bash
# Quick test run
!python train_stage1.py --num_epochs 5
```

---

## 📈 **Results Monitoring**

Kaggle will automatically save:
- Training curves in console output  
- Best model checkpoints in `/kaggle/working/checkpoints/`
- Feature files in `/kaggle/working/features_i3d/`
- Metrics logs for analysis

---

## 🎉 **Success Indicators**

✅ **Feature Extraction Complete**: Files in `/kaggle/working/features_i3d/`  
✅ **Stage 1 Training**: AUC > 0.7, checkpoint saved  
✅ **Stage 2 Training**: Improved accuracy, final model saved  
✅ **Pipeline Complete**: All metrics logged, models ready for inference  

---

## 📝 **Citation**

If you use this implementation, please cite:
```
@article{sultani2018real,
  title={Real-world anomaly detection in surveillance videos},
  author={...},
  journal={...},
  year={2018}
}
```

---

## 🔗 **Links**

- **Paper**: [2209.11477v1](https://arxiv.org/abs/2209.11477)
- **Original Dataset**: [RWF-2000](...)  
- **Pre-trained Models**: Available after training

**Happy Training on Kaggle! 🚀**