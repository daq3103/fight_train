"""
Complete pipeline runner for paper 2209.11477v1 methodology
Handles offline feature extraction, Stage 1 MIL training, and Stage 2 pseudo-label training
"""

import os
import argparse
import time
from datetime import datetime

from extract_i3d_features import extract_dataset_features
from train_stage1 import Stage1Trainer
from train_stage2 import Stage2Trainer
from data.stage1_dataset import create_stage1_dataloaders
from data.stage2_dataset import create_stage2_dataloaders
from config import Config


def setup_directories(config):
    """Setup required directories"""
    directories = [
        config.feature_dir,
        config.checkpoint_dir,
        os.path.join(config.checkpoint_dir, "stage1"),
        os.path.join(config.checkpoint_dir, "stage2"),
    ]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


def run_feature_extraction(config, overwrite=False):
    """Step 1: Extract I3D features from videos"""
    print("=" * 80)
    print("STEP 1: I3D Feature Extraction")
    print("=" * 80)

    if os.path.exists(config.feature_dir) and not overwrite:
        feature_files = len(
            [f for f in os.listdir(config.feature_dir) if f.endswith(".npy")]
        )
        if feature_files > 0:
            print(
                f"Found {feature_files} existing feature files in {config.feature_dir}. Skipping extraction (use --force_retrain to overwrite)."
            )
            return True

    print(f"Extracting features from videos in {config.data_root}...")
    print(f"Output directory: {config.feature_dir}")
    print(f"Target clips: {config.num_clips_stage1}")

    try:
        success = extract_dataset_features(
            data_root=config.data_root,
            output_dir=config.feature_dir,
            target_clips=config.num_clips_stage1,
        )

        if success:
            print("✓ Feature extraction completed successfully!")
            return True
        else:
            print("✗ Feature extraction failed!")
            return False

    except Exception as e:
        print(f"✗ Feature extraction failed with error: {str(e)}")
        return False


def run_stage1_training(config, force_retrain=False):
    """Step 2: Stage 1 MIL Training"""
    print("\n" + "=" * 80)
    print("STEP 2: Stage 1 MIL Training")
    print("=" * 80)

    stage1_checkpoint_dir = os.path.join(config.checkpoint_dir, "stage1")
    stage1_best_path = os.path.join(stage1_checkpoint_dir, "stage1_best.pth")

    # Check if Stage 1 is already trained
    if os.path.exists(stage1_best_path) and not force_retrain:
        print(f"Found existing Stage 1 checkpoint: {stage1_best_path}. Skipping training (use --force_retrain to retrain).")
        return stage1_best_path

    print(f"Starting Stage 1 MIL training...")
    print(f"Feature directory: {config.feature_dir}")
    print(f"Data root: {config.data_root}")
    print(f"Batch size: {config.batch_size}")
    print(f"Number of epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")

    try:
        # Create data loaders
        train_loader, test_loader = create_stage1_dataloaders(
            feature_dir=config.feature_dir,
            data_root=config.data_root,
            batch_size=config.batch_size,
            num_clips=config.num_clips_stage1,
            feature_dim=config.feature_dim,
            num_workers=config.num_workers,
        )

        print(f"Train batches: {len(train_loader)}")
        print(f"Test batches: {len(test_loader)}")

        if len(train_loader) == 0:
            print("✗ No training data found!")
            return None

        # Create trainer
        trainer = Stage1Trainer(config)

        # Start training
        trainer.train(
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=config.num_epochs,
            checkpoint_dir=stage1_checkpoint_dir,
        )

        if os.path.exists(stage1_best_path):
            print(f"✓ Stage 1 training completed successfully!")
            print(f"Best checkpoint: {stage1_best_path}")
            return stage1_best_path
        else:
            print("✗ Stage 1 training failed - no checkpoint saved!")
            return None

    except Exception as e:
        print(f"✗ Stage 1 training failed with error: {str(e)}")
        return None


def run_stage2_training(config, stage1_checkpoint_path, force_retrain=False):
    """Step 3: Stage 2 Pseudo-label Training"""
    print("\n" + "=" * 80)
    print("STEP 3: Stage 2 Pseudo-label Training")
    print("=" * 80)

    stage2_checkpoint_dir = os.path.join(config.checkpoint_dir, "stage2")
    stage2_best_path = os.path.join(stage2_checkpoint_dir, "stage2_best.pth")

    # Check if Stage 2 is already trained
    if os.path.exists(stage2_best_path) and not force_retrain:
        print(f"Found existing Stage 2 checkpoint: {stage2_best_path}. Skipping training (use --force_retrain to retrain).")
        return stage2_best_path

    print(f"Starting Stage 2 pseudo-label training...")
    print(f"Stage 1 checkpoint: {stage1_checkpoint_path}")
    print(f"Batch size: {config.batch_size_stage2}")
    print(f"Number of epochs: {config.num_epochs_stage2}")
    print(f"Learning rate: {config.learning_rate_stage2}")
    print(f"Freeze encoder: {config.freeze_encoder_stage2}")

    try:
        # Create data loaders
        train_loader, test_loader = create_stage2_dataloaders(
            feature_dir=config.feature_dir,
            data_root=config.data_root,
            batch_size=config.batch_size_stage2,
            feature_dim=config.feature_dim,
            max_clips=256,  # Allow variable length
            num_workers=config.num_workers,
        )

        print(f"Train batches: {len(train_loader)}")
        print(f"Test batches: {len(test_loader)}")

        if len(train_loader) == 0:
            print("✗ No training data found!")
            return None

        # Create trainer
        trainer = Stage2Trainer(config, stage1_checkpoint_path)

        # Start training
        trainer.train(
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=config.num_epochs_stage2,
            checkpoint_dir=stage2_checkpoint_dir,
        )

        if os.path.exists(stage2_best_path):
            print(f"✓ Stage 2 training completed successfully!")
            print(f"Best checkpoint: {stage2_best_path}")
            return stage2_best_path
        else:
            print("✗ Stage 2 training failed - no checkpoint saved!")
            return None

    except Exception as e:
        print(f"✗ Stage 2 training failed with error: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Full Pipeline for Paper 2209.11477v1")
    parser.add_argument(
        "--data_root",
        type=str,
        default="./new_youtube",
        help="Root directory of dataset",
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        default="./features_i3d",
        help="Directory to save I3D features",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints",
    )

    # Pipeline control
    parser.add_argument(
        "--skip_extraction", action="store_true", help="Skip feature extraction step"
    )
    parser.add_argument(
        "--skip_stage1", action="store_true", help="Skip Stage 1 training"
    )
    parser.add_argument(
        "--skip_stage2", action="store_true", help="Skip Stage 2 training"
    )
    parser.add_argument(
        "--force_retrain",
        action="store_true",
        help="Force retrain even if checkpoints exist",
    )

    # Training parameters
    parser.add_argument(
        "--stage1_epochs", type=int, default=100, help="Number of epochs for Stage 1"
    )
    parser.add_argument(
        "--stage2_epochs", type=int, default=50, help="Number of epochs for Stage 2"
    )
    parser.add_argument(
        "--stage1_batch_size", type=int, default=16, help="Batch size for Stage 1"
    )
    parser.add_argument(
        "--stage2_batch_size", type=int, default=4, help="Batch size for Stage 2"
    )
    parser.add_argument(
        "--stage1_lr", type=float, default=1e-3, help="Learning rate for Stage 1"
    )
    parser.add_argument(
        "--stage2_lr", type=float, default=1e-4, help="Learning rate for Stage 2"
    )

    args = parser.parse_args()

    # Create config
    config = Config()
    config.data_root = args.data_root
    config.feature_dir = args.feature_dir
    config.checkpoint_dir = args.checkpoint_dir
    config.num_epochs = args.stage1_epochs
    config.num_epochs_stage2 = args.stage2_epochs
    config.batch_size = args.stage1_batch_size
    config.batch_size_stage2 = args.stage2_batch_size
    config.learning_rate = args.stage1_lr
    config.learning_rate_stage2 = args.stage2_lr

    print("🚀 Starting Full Pipeline for Paper 2209.11477v1")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nConfiguration:")
    print(f"  Data root: {config.data_root}")
    print(f"  Feature directory: {config.feature_dir}")
    print(f"  Checkpoint directory: {config.checkpoint_dir}")
    print(f"  Stage 1 epochs: {config.num_epochs}")
    print(f"  Stage 2 epochs: {config.num_epochs_stage2}")
    print(f"  Stage 1 batch size: {config.batch_size}")
    print(f"  Stage 2 batch size: {config.batch_size_stage2}")
    print(f"  Stage 1 learning rate: {config.learning_rate}")
    print(f"  Stage 2 learning rate: {config.learning_rate_stage2}")

    # Setup directories
    setup_directories(config)

    # Pipeline execution
    pipeline_start_time = time.time()

    # Step 1: Feature Extraction
    if not args.skip_extraction:
        success = run_feature_extraction(config, overwrite=args.force_retrain)
        if not success:
            print("❌ Pipeline failed at feature extraction step!")
            return
    else:
        print("⏭️ Skipping feature extraction...")

    # Step 2: Stage 1 Training
    stage1_checkpoint_path = None
    if not args.skip_stage1:
        stage1_checkpoint_path = run_stage1_training(
            config, force_retrain=args.force_retrain
        )
        if stage1_checkpoint_path is None:
            print("❌ Pipeline failed at Stage 1 training!")
            return
    else:
        print("⏭️ Skipping Stage 1 training...")
        # Look for existing Stage 1 checkpoint
        stage1_best_path = os.path.join(
            config.checkpoint_dir, "stage1", "stage1_best.pth"
        )
        if os.path.exists(stage1_best_path):
            stage1_checkpoint_path = stage1_best_path
            print(f"Using existing Stage 1 checkpoint: {stage1_checkpoint_path}")
        else:
            print("❌ No Stage 1 checkpoint found and training skipped!")
            return

    # Step 3: Stage 2 Training
    if not args.skip_stage2:
        if stage1_checkpoint_path is None:
            print("❌ Stage 1 checkpoint required for Stage 2 training!")
            return

        stage2_checkpoint_path = run_stage2_training(
            config, stage1_checkpoint_path, force_retrain=args.force_retrain
        )
        if stage2_checkpoint_path is None:
            print("❌ Pipeline failed at Stage 2 training!")
            return
    else:
        print("⏭️ Skipping Stage 2 training...")

    # Pipeline completion
    pipeline_end_time = time.time()
    pipeline_duration = pipeline_end_time - pipeline_start_time

    print("\n" + "=" * 80)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Total time: {pipeline_duration/3600:.2f} hours")
    print(f"Final timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if stage1_checkpoint_path:
        print(f"Stage 1 checkpoint: {stage1_checkpoint_path}")

    stage2_best_path = os.path.join(config.checkpoint_dir, "stage2", "stage2_best.pth")
    if os.path.exists(stage2_best_path):
        print(f"Stage 2 checkpoint: {stage2_best_path}")

    print("\nYou can now use the trained models for inference!")


if __name__ == "__main__":
    main()
