"""
Kaggle Environment Setup Script for Fight Detection Pipeline
Automatically detects and configures paths for Kaggle environment
"""

import os
import sys
import shutil
from pathlib import Path


def setup_kaggle_environment():
    """Setup environment cho Kaggle"""
    print("🐾 Setting up Kaggle environment for Fight Detection...")

    # Check if running on Kaggle
    if "KAGGLE_KERNEL_RUN_TYPE" not in os.environ:
        print("⚠️  Not running on Kaggle, using local setup")
        return False

    # Kaggle paths
    kaggle_input = Path("/kaggle/input")
    kaggle_working = Path("/kaggle/working")

    print(f"📁 Kaggle input directory: {kaggle_input}")
    print(f"📁 Kaggle working directory: {kaggle_working}")

    # Find dataset
    dataset_paths = [
        kaggle_input / "fight-detection-dataset",
        kaggle_input / "new-youtube",
        kaggle_input / "youtube-fight-dataset",
    ]

    dataset_path = None
    for path in dataset_paths:
        if path.exists():
            dataset_path = path
            break

    if dataset_path is None:
        print("❌ Dataset not found in Kaggle input!")
        print("Available input datasets:")
        try:
            for item in kaggle_input.iterdir():
                print(f"  - {item.name}")
        except:
            print("  (Cannot list input directory)")
        return False

    print(f"✅ Found dataset at: {dataset_path}")

    # Setup directories
    feature_dir = kaggle_working / "features_i3d"
    checkpoint_dir = kaggle_working / "checkpoints"

    feature_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    print(f"📂 Feature directory: {feature_dir}")
    print(f"📂 Checkpoint directory: {checkpoint_dir}")

    # Create environment config
    env_config = {
        "ENVIRONMENT": "kaggle",
        "DATA_ROOT": str(dataset_path),
        "FEATURE_DIR": str(feature_dir),
        "CHECKPOINT_DIR": str(checkpoint_dir),
        "DEVICE": "cuda" if os.path.exists("/opt/bin/nvidia-smi") else "cpu",
    }

    # Save config
    config_file = kaggle_working / "kaggle_config.py"
    with open(config_file, "w") as f:
        f.write("# Auto-generated Kaggle configuration\\n")
        for key, value in env_config.items():
            f.write(f"{key} = '{value}'\\n")

    print("✅ Kaggle environment setup completed!")
    print("\\n📋 Configuration:")
    for key, value in env_config.items():
        print(f"  {key}: {value}")

    return True


def install_dependencies():
    """Install required packages for Kaggle"""
    print("\\n📦 Installing dependencies...")

    # Check GPU
    gpu_available = os.path.exists("/opt/bin/nvidia-smi")
    print(f"🔧 GPU available: {gpu_available}")

    if gpu_available:
        print("🚀 GPU detected - ready for training!")
    else:
        print("⚠️  No GPU detected - training will be slower")

    # Check PyTorch
    try:
        import torch

        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch not available!")
        return False

    # Check other dependencies
    dependencies = ["cv2", "numpy", "tqdm", "PIL"]
    for dep in dependencies:
        try:
            __import__(dep if dep != "PIL" else "PIL.Image")
            print(f"✅ {dep} available")
        except ImportError:
            print(f"❌ {dep} not available!")

    return True


def create_kaggle_notebook_commands():
    """Create ready-to-use commands for Kaggle notebook"""

    commands = {
        "Setup": [
            "import sys",
            "sys.path.append('/kaggle/working')",
            "exec(open('kaggle_setup.py').read())",
        ],
        "Feature Extraction": ["python extract_i3d_features.py --environment kaggle"],
        "Stage 1 Training": [
            "python train_stage1.py --feature_dir /kaggle/working/features_i3d --data_root /kaggle/input/fight-detection-dataset --batch_size 8 --num_epochs 50"
        ],
        "Stage 2 Training": [
            "python train_stage2.py --feature_dir /kaggle/working/features_i3d --stage1_checkpoint /kaggle/working/checkpoints/stage1/stage1_best.pth --batch_size 4 --num_epochs 30"
        ],
        "Full Pipeline": [
            "python run_pipeline.py --data_root /kaggle/input/fight-detection-dataset --feature_dir /kaggle/working/features_i3d --checkpoint_dir /kaggle/working/checkpoints"
        ],
    }

    # Save commands to file
    commands_file = Path("/kaggle/working/kaggle_commands.txt")
    with open(commands_file, "w") as f:
        f.write("# Kaggle Commands for Fight Detection Pipeline\\n")
        f.write("# Copy and paste these commands in Kaggle notebook cells\\n\\n")

        for section, cmds in commands.items():
            f.write(f"## {section}\\n")
            for cmd in cmds:
                f.write(f"{cmd}\\n")
            f.write("\\n")

    print(f"\\n📝 Commands saved to: {commands_file}")
    return commands


def main():
    """Main setup function"""
    print("🎯 Kaggle Setup for Fight Detection Pipeline")
    print("=" * 50)

    # Setup environment
    env_ok = setup_kaggle_environment()

    # Install dependencies
    deps_ok = install_dependencies()

    # Create commands
    if env_ok:
        commands = create_kaggle_notebook_commands()

    print("\\n" + "=" * 50)
    if env_ok and deps_ok:
        print("🎉 Kaggle setup completed successfully!")
        print("\\n🚀 Ready to run Fight Detection Pipeline on Kaggle!")
        print("\\n📖 Next steps:")
        print("  1. Check kaggle_commands.txt for ready commands")
        print("  2. Run feature extraction first")
        print("  3. Train Stage 1 and Stage 2 models")
        print("  4. Or run full pipeline at once")
    else:
        print("❌ Setup encountered issues!")
        print("\\n🔧 Troubleshooting:")
        print("  1. Ensure dataset is uploaded to Kaggle")
        print("  2. Check dataset naming (should contain train_data, test_data)")
        print("  3. Verify all dependencies are available")

    return env_ok and deps_ok


if __name__ == "__main__":
    main()
