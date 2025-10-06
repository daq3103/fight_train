# YouTube Fight Detection Dataset

## Tổng quan

Dự án này triển khai hệ thống phát hiện đánh nhau trong video YouTube sử dụng:
- **R2Plus1D backbone**: Trích xuất features từ video
- **Multiple Temporal Network (MTN)**: Phân tích temporal patterns ở nhiều scale
- **Anomaly Score Generator**: Sinh anomaly scores và classification logits

## Cấu trúc Dataset

```
new_youtube/
├── fight_vid.txt          # Danh sách video đánh nhau
├── nofight_vid.txt        # Danh sách video không đánh nhau  
├── train_data/            # Video training
│   ├── f_*.mp4           # Fight videos
│   └── nof_*.mp4         # No-fight videos
├── test_data/             # Video testing
│   ├── f_*.mp4           # Fight videos
│   └── nof_*.mp4         # No-fight videos
└── meta_data/             # Metadata và thông tin bổ sung
```

## Dataset Features

- **Total videos**: ~2,370 videos
- **Fight videos**: ~1,190 videos (prefix `f_`)
- **No-fight videos**: ~1,180 videos (prefix `nof_`)
- **Classes**: 2 (Fight=1, No-Fight=0)
- **Format**: MP4 videos với độ dài khác nhau

## Cài đặt

### 1. Dependencies

```bash
pip install torch torchvision opencv-python pillow numpy tqdm
```

### 2. Kiểm tra Dataset

```bash
python test_dataset.py
```

### 3. Training đơn giản

```bash
python simple_train.py
```

### 4. Training đầy đủ

```bash
python train_youtube.py --data_path ./new_youtube --batch_size 4 --epochs 50
```

## Sử dụng Dataset

### 1. Basic Usage

```python
from data.youtube_dataset import YouTubeFightDataset, get_transforms

# Tạo dataset
transform = get_transforms('train')
dataset = YouTubeFightDataset(
    data_root='./new_youtube',
    mode='train',
    sequence_length=16,
    transform=transform
)

# Lấy sample
video, targets = dataset[0]
print(f"Video shape: {video.shape}")  # (C, T, H, W)
print(f"Label: {targets['labels']}")  # 0 hoặc 1
```

### 2. DataLoader

```python
from data.youtube_dataset import create_dataloaders

train_loader, test_loader = create_dataloaders(
    data_root='./new_youtube',
    batch_size=4,
    sequence_length=16,
    input_size=224,
    num_workers=2
)

for videos, targets in train_loader:
    print(f"Batch videos: {videos.shape}")  # (B, C, T, H, W)
    print(f"Batch labels: {targets['labels']}")  # (B,)
    break
```

### 3. Model Training

```python
from models.complete_model import R3D_MTN_AnomalyDetector

model = R3D_MTN_AnomalyDetector(
    num_classes=2,
    r3d_pretrained=True,
    mtn_hidden_dim=512,
    use_mil=True
)

# Forward pass
results = model(videos, return_classification=True)
print(f"Anomaly scores: {results['anomaly_scores'].shape}")
print(f"Classification logits: {results['classification_logits'].shape}")
```

## Model Architecture

### 1. R2Plus1D Backbone
- Pretrained trên Kinetics-400
- Trích xuất multi-scale features: low_level, mid_level, high_level, final_level
- Option để freeze early layers

### 2. Multiple Temporal Network (MTN)
- Temporal Pyramid Module với scales [1, 2, 4, 8]
- Đồng bộ features về cùng temporal dimension
- Temporal attention mechanism

### 3. Anomaly Score Generator
- MIL (Multiple Instance Learning) aggregation
- Frame-level và video-level anomaly scores
- Classification head cho supervised learning

## Training Parameters

```python
# Recommended settings
--batch_size 4              # GPU memory constraints
--sequence_length 16        # 16 frames per clip
--learning_rate 1e-4        # Conservative learning rate
--epochs 50                 # Sufficient for convergence
--freeze_r3d_layers 2       # Freeze early R2Plus1D layers
--mtn_hidden_dim 512        # MTN feature dimension
```

## Performance Tips

### 1. Memory Optimization
- Giảm `batch_size` nếu GPU memory không đủ
- Sử dụng `num_workers=0` khi debugging
- Set `pin_memory=True` cho GPU training

### 2. Training Stability
- Gradient clipping: `max_norm=1.0`
- Learning rate scheduling: StepLR
- Freeze backbone layers ban đầu

### 3. Data Augmentation
- RandomHorizontalFlip cho training
- ColorJitter để tăng tính robust
- Normalize với ImageNet stats

## Evaluation Metrics

- **Accuracy**: Classification accuracy
- **Loss Components**:
  - Anomaly loss: MSE giữa predicted và ground truth anomaly scores
  - Classification loss: CrossEntropy cho fight/no-fight
  - Smoothness loss: Temporal smoothness regularization

## Troubleshooting

### 1. Dataset không load được
- Kiểm tra path: `./new_youtube`
- Kiểm tra file tồn tại: `fight_vid.txt`, `nofight_vid.txt`
- Kiểm tra video files trong `train_data/` và `test_data/`

### 2. CUDA out of memory
- Giảm `batch_size` từ 4 xuống 2 hoặc 1
- Giảm `sequence_length` từ 16 xuống 8
- Giảm `input_size` từ 224 xuống 112

### 3. Model training chậm
- Increase `num_workers` để tăng tốc data loading
- Sử dụng mixed precision training
- Freeze nhiều backbone layers hơn

## Files Structure

```
I3D/
├── data/
│   ├── youtube_dataset.py     # YouTube dataset loader
│   └── dataset.py             # Generic dataset (legacy)
├── models/
│   ├── complete_model.py      # Complete model definition
│   ├── r3d_backbone.py        # R2Plus1D backbone
│   └── mtn_anomaly.py         # MTN + Anomaly modules
├── simple_train.py            # Simple training script
├── train_youtube.py           # Full training script
├── test_dataset.py            # Dataset testing script
└── new_youtube/               # Dataset folder
```

## Next Steps

1. **Hyperparameter tuning**: Grid search cho learning rate, batch size
2. **Data augmentation**: Thêm temporal augmentation
3. **Model improvements**: Ensemble methods, attention mechanisms
4. **Evaluation**: Detailed metrics, confusion matrix, ROC curves
5. **Deployment**: Model optimization, inference pipeline