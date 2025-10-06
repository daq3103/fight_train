import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights


class R2Plus1DBackbone(nn.Module):
    """
    R(2+1)D Feature Extractor - simplified version
    Compatible with I3D interface - returns same multi-scale features
    """
    
    def __init__(self, pretrained=True, freeze_backbone=False):
        super(R2Plus1DBackbone, self).__init__()
        
        # Load pretrained R(2+1)D model
        weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        self.backbone = r2plus1d_18(weights=weights)
        
        self.backbone.fc = nn.Identity() 

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.features = {}
        self._register_hooks()
        
        self.feature_processors = nn.ModuleDict({
            'low_level_proc': nn.Conv3d(64, 192, kernel_size=1),    
            'mid_level_proc': nn.Conv3d(128, 480, kernel_size=1),   
            'high_level_proc': nn.Conv3d(256, 832, kernel_size=1), 
            'final_level_proc': nn.Conv3d(512, 1024, kernel_size=1) 
        })
    
    def _register_hooks(self): 
        """Register forward hooks to extract multi-scale features"""
        
        def hook_fn(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook
        
        # Register hooks at ResNet layers
        if hasattr(self.backbone, 'layer1'):
            self.backbone.layer1.register_forward_hook(hook_fn('layer1'))
        if hasattr(self.backbone, 'layer2'):
            self.backbone.layer2.register_forward_hook(hook_fn('layer2'))
        if hasattr(self.backbone, 'layer3'):
            self.backbone.layer3.register_forward_hook(hook_fn('layer3'))
        if hasattr(self.backbone, 'layer4'):
            self.backbone.layer4.register_forward_hook(hook_fn('layer4'))
    
    def forward(self, x):

        self.features = {}
        
        global_features = self.backbone(x)  
        
        processed_features = {}
        
        if 'layer1' in self.features:
            low_feat = self.feature_processors['low_level_proc'](self.features['layer1'])
            processed_features['low_level'] = low_feat
        
        if 'layer2' in self.features:
            mid_feat = self.feature_processors['mid_level_proc'](self.features['layer2'])
            processed_features['mid_level'] = mid_feat
        
        if 'layer3' in self.features:
            high_feat = self.feature_processors['high_level_proc'](self.features['layer3'])
            processed_features['high_level'] = high_feat
        
        if 'layer4' in self.features:
            final_feat = self.feature_processors['final_level_proc'](self.features['layer4'])
            processed_features['final_level'] = final_feat
        
        processed_features['global_features'] = torch.zeros(
            global_features.size(0), 1024, 1, device=global_features.device
        )
        processed_features['global_features'][:, :512, 0] = global_features
        
        return processed_features



if __name__ == "__main__":
    # Test R(2+1)D model only
    x = torch.randn(2, 3, 32, 112, 112)  

    model = R2Plus1DBackbone(pretrained=True, freeze_backbone=False)
    model.eval()
    
    with torch.no_grad():
        features = model(x)
    
    # Print results
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    for feat_name, feat in features.items():
        print(f"    {feat_name}: {feat.shape}")
