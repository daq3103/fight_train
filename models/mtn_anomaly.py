"""
triển khai Mô-đun Mạng Thời gian Đa cấp (MTN) và Bộ tạo Điểm Bất thường với Học Đa Thể (MIL) 
- MTN: Giúp mô hình nắm bắt bất thường ở nhiều thang thời gian → mạnh trong việc phát hiện hành vi có độ dài khác nhau
- MIL anomaly generator: Cho phép huấn luyện với weak labels (clip-level)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TemporalPyramidModule(nn.Module):
    """
    input: (batch, channels, time, height, width)
    output: (batch, out_channels, time, height, width)
    out_channels = hidden_dim (512) 
    """
    
    def __init__(self, in_channels, out_channels, temporal_scales=[1, 2, 4, 8]):
        super(TemporalPyramidModule, self).__init__()
        self.temporal_scales = temporal_scales
        self.branches = nn.ModuleList()

        for scale in temporal_scales:
            branch = nn.Sequential(
                nn.Conv3d(in_channels, out_channels // len(temporal_scales), 
                         kernel_size=(scale * 2 + 1, 1, 1), 
                         padding=(scale, 0, 0)),
                nn.BatchNorm3d(out_channels // len(temporal_scales)), 
                nn.ReLU(inplace=True)
            ) 
            self.branches.append(branch) 
        
        # Fusion layer 
        self.fusion = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, channels, time, height, width)
        Returns:
            Multi-scale temporal features
        """
        branch_outputs = [] # lưu trữ đầu ra từ mỗi nhánh xử lý song song
        
        for branch in self.branches:
            branch_out = branch(x)
            branch_outputs.append(branch_out)
        
        # Concatenate all branches
        combined = torch.cat(branch_outputs, dim=1)
        
        # Fusion
        output = self.fusion(combined)
        return output


class MultipleTemporalNetwork(nn.Module):
    """
    input: Dictionary of features from I3D backbone
        - low_level: (B, C1, T, H, W)
        - mid_level: (B, C2, T, H, W)
        - high_level: (B, C3, T, H, W)
        - final_level: (B, C4, T, H, W)
    output: Enhanced temporal features (B, hidden_dim, T)
    1. Temporal Pyramid Modules: Capture multi-scale temporal patterns
    2. Feature Fusion: Combine features across different levels
    """
    
    def __init__(self, feature_channels_dict, hidden_dim=512):
        super(MultipleTemporalNetwork, self).__init__()
        
        self.feature_channels = feature_channels_dict
        self.hidden_dim = hidden_dim
        
        # Temporal pyramid modules for different feature levels
        self.temporal_pyramids = nn.ModuleDict()
        
        for level, channels in feature_channels_dict.items():
            self.temporal_pyramids[level] = TemporalPyramidModule(
                in_channels=channels,
                out_channels=hidden_dim,
                temporal_scales=[1, 2]
            )
        
        # Feature fusion across different levels
        total_features = len(feature_channels_dict) * hidden_dim 
        self.level_fusion = nn.Sequential(
            nn.Conv3d(total_features, hidden_dim, kernel_size=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((None, 1, 1))  
        )  # bỏ qua không gian, giữ lại thời gian 
        
        # Temporal attention mechanism
        self.temporal_attention = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, features_dict):
        valid_features = {}
        for level, features in features_dict.items():
            if level in self.temporal_pyramids and features.dim() == 5:
                valid_features[level] = features
    
        temporal_sizes = [features.size(2) for features in valid_features.values()]
        target_t = min(temporal_sizes)  # 4 instead of max
        
        pyramid_outputs = []
        for level, features in valid_features.items():
            if features.size(2) != target_t:
                features = F.interpolate(features, 
                                   size=(target_t, features.size(3), features.size(4)),
                                   mode='trilinear', align_corners=False)
        
            # Qua temporal pyramid với T=4
            pyramid_out = self.temporal_pyramids[level](features)
            pyramid_outputs.append(pyramid_out)
        
        # Đảm bảo tất cả các đặc trưng có cùng kích thước không gian 
        min_h = min([feat.size(3) for feat in pyramid_outputs]) 
        min_w = min([feat.size(4) for feat in pyramid_outputs]) 
        
        resized_outputs = []
        for feat in pyramid_outputs:
            if feat.size(3) != min_h or feat.size(4) != min_w:
                feat = F.interpolate(feat, size=(feat.size(2), min_h, min_w), 
                                   mode='trilinear', align_corners=False)
            resized_outputs.append(feat)
        
        # Concatenate and fuse
        combined_features = torch.cat(resized_outputs, dim=1)
        fused_features = self.level_fusion(combined_features)  # (B, hidden_dim, T, 1, 1)
        
        # Remove spatial dimensions
        temporal_features = fused_features.squeeze(-1).squeeze(-1)  # (B, hidden_dim, T)
        
        # Apply temporal attention
        attention_weights = self.temporal_attention(temporal_features)  # (B, 1, T)
        attended_features = temporal_features * attention_weights  # (B, hidden_dim, T)
        
        return attended_features


class AnomalyScoreGenerator(nn.Module):
    """MLP-based Anomaly Score Generator with Multiple Instance Learning"""
    
    def __init__(self, input_dim=512, hidden_dims=[256, 128], num_classes=2, 
                 use_mil=True, dropout_prob=0.3):
        super(AnomalyScoreGenerator, self).__init__()
        
        self.input_dim = input_dim 
        self.use_mil = use_mil
        
        # Feature transformation layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob)
            ])
            prev_dim = hidden_dim
        
        self.feature_transform = nn.Sequential(*layers) # (B*T, hidden_dim) B batch size, T time steps 
        
        # Anomaly scoring layers
        self.anomaly_scorer = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output anomaly scores in [0, 1]
        )
        
        # Classification head (for supervised learning)
        self.classifier = nn.Sequential(
            nn.Linear(prev_dim, num_classes)
        )
        
        if use_mil:
            # Multiple Instance Learning aggregation
            self.mil_attention = nn.Sequential(
                nn.Linear(prev_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )

    def forward(self, temporal_features, return_scores=True, return_classification=True):
        
        B, C, T = temporal_features.shape
        
        # Reshape for processing: (B*T, C)
        features_reshaped = temporal_features.permute(0, 2, 1).contiguous()  # (B, T, C)
        features_flat = features_reshaped.view(B * T, C)  # (B*T, C)
        
        # Feature transformation
        transformed_features = self.feature_transform(features_flat)  # (B*T, hidden_dim)
        
        results = {}
        
        if return_scores:
            # Frame-level anomaly scores
            frame_scores = self.anomaly_scorer(transformed_features)  # (B*T, 1)
            frame_scores = frame_scores.view(B, T)  # (B, T)
            results['anomaly_scores'] = frame_scores
            
            # Video-level aggregation
            if self.use_mil:
                # MIL-based aggregation with attention
                mil_features = transformed_features.view(B, T, -1)  # (B, T, hidden_dim)
                
                # Compute attention weights
                attention_scores = self.mil_attention(mil_features)  # (B, T, 1)
                attention_weights = F.softmax(attention_scores, dim=1)  # (B, T, 1)
                
                # Weighted aggregation
                video_features = torch.sum(mil_features * attention_weights, dim=1)  # (B, hidden_dim)
                video_score = self.anomaly_scorer(video_features).squeeze(1)  # (B,)
            else:
                # Simple max pooling
                video_score = torch.max(frame_scores, dim=1)[0]  # (B,)
            
            results['video_score'] = video_score
            results['attention_weights'] = attention_weights.squeeze(-1) if self.use_mil else None
        
        if return_classification:
            # Classification for supervised learning
            if self.use_mil and 'video_features' in locals():
                # Use MIL aggregated features
                classification_logits = self.classifier(video_features)
            else:
                # Use average pooled features
                avg_features = torch.mean(transformed_features.view(B, T, -1), dim=1)
                classification_logits = self.classifier(avg_features)
            
            results['classification_logits'] = classification_logits
        
        return results


class AnomalyDetectionLoss(nn.Module):
    """Combined loss for anomaly detection and classification"""
    
    def __init__(self, anomaly_weight=1.0, classification_weight=1.0, 
                 smoothness_weight=0.1):
        super(AnomalyDetectionLoss, self).__init__()
        
        self.anomaly_weight = anomaly_weight 
        self.classification_weight = classification_weight
        self.smoothness_weight = smoothness_weight

        self.bce_loss = nn.BCELoss() # dùng cho điểm bất thường (0-1) 
        self.ce_loss = nn.CrossEntropyLoss() # dùng cho phân loại (logits) 

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Dictionary from AnomalyScoreGenerator
            targets: Dictionary containing:
                - labels: (B,) video-level labels (0: normal, 1: anomaly)
                - frame_labels: (B, T) frame-level labels (optional)
        
        Returns:
            total_loss, loss_dict
        """
        total_loss = 0.0
        loss_dict = {}
        
        # Classification loss
        if 'classification_logits' in predictions and 'labels' in targets:
            cls_loss = self.ce_loss(predictions['classification_logits'], targets['labels'])
            total_loss += self.classification_weight * cls_loss
            loss_dict['classification_loss'] = cls_loss.item()
        
        # Anomaly detection loss
        if 'video_score' in predictions and 'labels' in targets:
            # Convert labels to float for BCE
            anomaly_labels = targets['labels'].float()
            anomaly_loss = self.bce_loss(predictions['video_score'], anomaly_labels)
            total_loss += self.anomaly_weight * anomaly_loss
            loss_dict['anomaly_loss'] = anomaly_loss.item()
        
        # Temporal smoothness loss (encourage smooth anomaly scores)
        if 'anomaly_scores' in predictions and self.smoothness_weight > 0:
            frame_scores = predictions['anomaly_scores']  # (B, T)
            smoothness_loss = torch.mean(torch.abs(frame_scores[:, 1:] - frame_scores[:, :-1]))
            total_loss += self.smoothness_weight * smoothness_loss
            loss_dict['smoothness_loss'] = smoothness_loss.item()
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict

