"""
Configuration file for I3D + MTN Anomaly Detection following paper 2209.11477v1
Auto-detects environment and sets appropriate paths
"""

import os


def detect_environment():
    """Auto-detect running environment"""
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "kaggle"
    elif "COLAB_GPU" in os.environ:
        return "colab"
    else:
        return "local"


def get_default_paths(environment=None):
    """Get default paths based on environment"""
    if environment is None:
        environment = detect_environment()

    if environment == "kaggle":
        return {
            "data_root": "/kaggle/input",
            "feature_dir": "/kaggle/working/features_i3d",
            "checkpoint_dir": "/kaggle/working/checkpoints",
        }
    elif environment == "colab":
        return {
            "data_root": "/content/new_youtube",
            "feature_dir": "/content/drive/MyDrive/fight_detect/features_i3d",
            "checkpoint_dir": "/content/drive/MyDrive/fight_detect/checkpoints",
        }
    else:  # local
        return {
            "data_root": "./new_youtube",
            "feature_dir": "./features_i3d",
            "checkpoint_dir": "./checkpoints",
        }


class Config:
    """Base configuration class for paper 2209.11477v1 methodology"""

    def __init__(self, environment=None):
        # Auto-detect environment and set paths
        self.environment = environment or detect_environment()
        default_paths = get_default_paths(self.environment)

        # Data paths (can be overridden)
        self.data_root = default_paths["data_root"]
        self.feature_dir = default_paths["feature_dir"]
        self.checkpoint_dir = default_paths["checkpoint_dir"]

    # Model parameters
    backbone_type = "r2plus1d_18"  # For I3D feature extraction
    i3d_pretrained = True
    feature_dim = 1024  # R(2+1)D pooled features used in dataset

    # MTN parameters
    hidden_dim = 512
    num_layers = 2
    dropout = 0.3
    use_attention = True
    num_classes = 2  # Fight vs No-fight

    # Video preprocessing parameters (for feature extraction)
    clip_length_frames = 32  # T=32 frames per clip (paper specification)
    frame_size = (224, 224)  # Input size for I3D
    temporal_stride = 16  # Stride between clips (overlap)
    frame_rate = 30

    # Stage 1 MIL parameters
    num_clips_stage1 = 32  # N=32 clips per video after uniform grouping
    batch_size = 16
    learning_rate = 1e-3
    weight_decay = 1e-4
    num_epochs = 100

    # Stage 2 parameters
    batch_size_stage2 = 4  # Smaller for variable-length videos
    learning_rate_stage2 = 1e-4
    num_epochs_stage2 = 50
    freeze_encoder_stage2 = True  # Freeze encoder in Stage 2

    # Pseudo-label generation
    pseudo_label_strategy = "threshold"  # "threshold", "top_k", "soft"
    pseudo_label_threshold = 0.5
    pseudo_label_top_k = None  # Will be set to 30% of clips

    # Loss function weights (from paper)
    lambda_ranking = 1.0  # Ranking loss weight
    lambda_sparsity = 8e-3  # Sparsity loss weight (paper value)
    lambda_smoothness = 8e-4  # Smoothness loss weight (paper value)
    ranking_margin = 1.0  # Margin for ranking loss

    # Optimizer settings
    step_size = 30  # LR scheduler step size
    gamma = 0.5  # LR decay factor

    # Data loading
    num_workers = 2
    pin_memory = True

    # Evaluation thresholds
    anomaly_threshold = 0.5

    # Reproducibility
    random_seed = 42


class FightDetectionConfig(Config):
    """Configuration for fight detection with full MTN architecture"""

    # Task-specific parameters
    TASK_NAME = "fight_detection"
    NUM_CLASSES = 2  # Fight vs Non-fight
    BACKBONE_TYPE = "r2plus1d_18"

    # Data paths
    DATA_ROOT = "new_youtube"

    # Video parameters for MTN
    CLIP_LENGTH = 32  # Frames per clip
    TEMPORAL_STRIDE = 2
    FRAME_SIZE = (224, 224)

    # MTN architecture parameters
    MTN_HIDDEN_DIM = 512
    MTN_NUM_SCALES = 2
    MTN_DROPOUT = 0.3

    # Training parameters
    BATCH_SIZE = 2  # Smaller for video processing
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # Scheduler
    LR_STEP_SIZE = 10
    LR_GAMMA = 0.5
    EARLY_STOPPING_PATIENCE = 10

    # Loss weights
    ALPHA_ANOMALY = 1.0
    ALPHA_TEMPORAL = 0.1
    ALPHA_SPARSITY = 0.01


class ViolenceDetectionConfig(Config):
    """Configuration for general violence detection"""

    # Task-specific parameters
    TASK_NAME = "violence_detection"
    NUM_CLASSES = 1  # Binary anomaly detection

    # Video parameters
    CLIP_LENGTH = 128  # Longer clips for context
    FRAME_SIZE = (256, 256)  # Higher resolution

    # Model parameters
    I3D_DROPOUT = 0.3
    MTN_HIDDEN_DIM = 1024
    MTN_NUM_SCALES = 4

    # Training parameters
    BATCH_SIZE = 2  # Smaller batch due to larger clips
    NUM_EPOCHS = 200
    LEARNING_RATE = 1e-5

    # Enhanced temporal modeling
    ALPHA_TEMPORAL = 0.2


class DebugConfig(Config):
    """Configuration for debugging and development"""

    # Small data for quick testing
    CLIP_LENGTH = 16
    BATCH_SIZE = 2
    NUM_EPOCHS = 5
    FRAME_SIZE = (112, 112)

    # Frequent logging
    LOG_FREQUENCY = 10
    EVAL_FREQUENCY = 1
    SAVE_FREQUENCY = 2

    # Simple model
    MTN_HIDDEN_DIM = 128
    MTN_NUM_SCALES = 2

    # Fast training
    NUM_WORKERS = 0

    # Override device for CPU testing
    DEVICE = "cpu"


def get_config(config_name="default"):
    """Get configuration by name"""

    config_map = {
        "default": Config,
        "fight": FightDetectionConfig,
        "violence": ViolenceDetectionConfig,
        "debug": DebugConfig,
    }

    if config_name not in config_map:
        raise ValueError(
            f"Unknown config: {config_name}. Available: {list(config_map.keys())}"
        )

    return config_map[config_name]()


def create_experiment_config(base_config="default", **kwargs):
    """Create custom configuration by overriding base config"""

    config = get_config(base_config)

    # Override with provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key.upper()):
            setattr(config, key.upper(), value)
        else:
            print(f"Warning: Unknown config parameter {key}")

    return config


if __name__ == "__main__":
    # Test configurations

    print("=== Default Config ===")
    default_config = get_config("default")
    print(f"Clip length: {default_config.CLIP_LENGTH}")
    print(f"Batch size: {default_config.BATCH_SIZE}")
    print(f"Learning rate: {default_config.LEARNING_RATE}")

    print("\n=== Fight Detection Config ===")
    fight_config = get_config("fight")
    print(f"Clip length: {fight_config.CLIP_LENGTH}")
    print(f"Batch size: {fight_config.BATCH_SIZE}")
    print(f"Learning rate: {fight_config.LEARNING_RATE}")

    print("\n=== Debug Config ===")
    debug_config = get_config("debug")
    print(f"Clip length: {debug_config.CLIP_LENGTH}")
    print(f"Batch size: {debug_config.BATCH_SIZE}")
    print(f"Device: {debug_config.DEVICE}")

    print("\n=== Custom Config ===")
    custom_config = create_experiment_config(
        base_config="fight", clip_length=48, batch_size=16, learning_rate=1e-3
    )
    print(f"Clip length: {custom_config.CLIP_LENGTH}")
    print(f"Batch size: {custom_config.BATCH_SIZE}")
    print(f"Learning rate: {custom_config.LEARNING_RATE}")
