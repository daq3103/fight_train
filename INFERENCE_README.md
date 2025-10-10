# 🥊 Fight Detection Inference Guide

Hướng dẫn sử dụng model đã train để detect fight trong video.

## 📁 Files được tạo

1. **`infer.py`** - Script inference đầy đủ với nhiều tùy chọn
2. **`quick_infer.py`** - Script inference nhanh, đơn giản
3. **`check_checkpoints.py`** - Script kiểm tra checkpoint có sẵn

## 🚀 Cách sử dụng

### 1. Kiểm tra checkpoint có sẵn

```bash
python check_checkpoints.py
```

### 2. Inference nhanh (đơn giản nhất)

```bash
python quick_infer.py
```

Hoặc với video khác:
```bash
python quick_infer.py  # Sẽ dùng test.mp4 mặc định
```

### 3. Inference đầy đủ

```bash
# Sử dụng checkpoint mặc định
python infer.py --video_path test.mp4

# Sử dụng checkpoint cụ thể
python infer.py --video_path test.mp4 --checkpoint_path ./checkpoints/stage1_best.pth

# Thay đổi threshold
python infer.py --video_path test.mp4 --threshold 0.3

# Lưu kết quả ra file
python infer.py --video_path test.mp4 --output_path results.json

# Sử dụng CPU
python infer.py --video_path test.mp4 --device cpu
```

## 📊 Kết quả mẫu

```
============================================================
🥊 FIGHT DETECTION RESULTS
============================================================
Video: test.mp4
Prediction: FIGHT
Confidence: 0.8234
Threshold: 0.5
Inference time: 2.45s
Number of clips: 32
Fight clips: 18/32 (56.3%)

Top 5 most suspicious clips:
  1. Clip 15: 0.9234
  2. Clip 8: 0.8765
  3. Clip 22: 0.8123
  4. Clip 3: 0.7891
  5. Clip 28: 0.7654
============================================================
```

## 🔧 Tham số

### infer.py
- `--video_path`: Đường dẫn video (mặc định: test.mp4)
- `--checkpoint_path`: Đường dẫn checkpoint (mặc định: ./checkpoints/stage1_best.pth)
- `--threshold`: Ngưỡng phân loại (mặc định: 0.5)
- `--output_path`: Lưu kết quả ra file JSON
- `--device`: Device sử dụng (cuda/cpu, auto-detect nếu không chỉ định)

### quick_infer.py
- Sử dụng tham số mặc định
- Chỉ cần đặt video `test.mp4` trong thư mục gốc

## 📋 Yêu cầu

1. **Video input**: File video (mp4, avi, etc.)
2. **Checkpoint**: Model đã train (file .pth)
3. **Dependencies**: 
   - torch
   - opencv-python
   - numpy
   - PIL

## 🎯 Model Performance

- **Best AUC**: 0.8061 (epoch 7)
- **Final AUC**: 0.7704
- **Accuracy**: 0.6786
- **Input**: 32 clips per video (uniform segmentation)
- **Feature**: I3D features (1024-dim)

## 🐛 Troubleshooting

### Lỗi "Checkpoint not found"
```bash
# Kiểm tra checkpoint có sẵn
python check_checkpoints.py

# Hoặc train model mới
python train_stage1.py
```

### Lỗi "Video not found"
- Đảm bảo file `test.mp4` tồn tại
- Hoặc chỉ định đường dẫn đúng: `--video_path /path/to/your/video.mp4`

### Lỗi CUDA
```bash
# Sử dụng CPU
python infer.py --device cpu
```

### Lỗi feature extraction
- Đảm bảo video có thể đọc được
- Kiểm tra format video (mp4, avi, etc.)

## 📈 Hiểu kết quả

- **Video Score**: Điểm confidence cao nhất trong tất cả clips
- **Video Prediction**: FIGHT/NO-FIGHT dựa trên threshold
- **Clip Scores**: Điểm confidence cho từng clip (32 clips)
- **Fight Clips**: Số clips được phân loại là fight

## 🔄 Workflow

1. **Extract Features**: Video → 32 clips → I3D features
2. **Model Inference**: Features → Model → Logits
3. **Post-processing**: Logits → Probabilities → Predictions
4. **Results**: Video-level + Clip-level predictions

## 💡 Tips

- **Threshold 0.3-0.7**: Điều chỉnh độ nhạy
- **Top clips**: Xem clips nào model nghĩ là fight
- **Inference time**: ~2-3s cho video thông thường
- **GPU**: Nhanh hơn CPU 3-5x
