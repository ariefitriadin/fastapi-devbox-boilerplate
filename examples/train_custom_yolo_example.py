"""
YOLO Custom Training Example Script

This script demonstrates how to train YOLO on a custom dataset.
Includes dataset preparation, training, evaluation, and deployment.

Requirements:
    - Annotated dataset in YOLO format
    - GPU recommended (but not required)
    - Ultralytics YOLO installed

Usage:
    # Quick start (with prepared dataset)
    python examples/train_custom_yolo_example.py --data dataset/data.yaml

    # Full pipeline (from raw images)
    python examples/train_custom_yolo_example.py --mode full --images dataset/raw_images
"""

import argparse
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import cv2
    import torch
    from ultralytics import YOLO
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("📦 Install with: poetry add ultralytics torch opencv-python")
    sys.exit(1)


def organize_dataset(
    source_images_dir,
    source_labels_dir,
    output_dir,
    class_names,
    train_split=0.8,
    val_split=0.15,
    test_split=0.05,
):
    """
    Organize raw dataset into YOLO training structure

    Args:
        source_images_dir: Directory with all images
        source_labels_dir: Directory with all labels (.txt files)
        output_dir: Output directory for organized dataset
        class_names: List of class names in order
        train_split: Percentage for training
        val_split: Percentage for validation
        test_split: Percentage for testing
    """
    print("\n" + "=" * 60)
    print("📦 ORGANIZING DATASET")
    print("=" * 60)

    source_images = Path(source_images_dir)
    source_labels = Path(source_labels_dir)
    output = Path(output_dir)

    # Validate splits
    assert abs(train_split + val_split + test_split - 1.0) < 0.001, (
        "Splits must sum to 1.0"
    )

    # Create directory structure
    for split in ["train", "val", "test"]:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Get all images with labels
    image_files = []
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        for img_file in source_images.glob(f"*{ext}"):
            label_file = source_labels / f"{img_file.stem}.txt"
            if label_file.exists():
                image_files.append(img_file)
            else:
                print(f"⚠️  No label for: {img_file.name}")

    if not image_files:
        print("❌ No images with labels found!")
        return False

    print(f"✅ Found {len(image_files)} images with labels")

    # Shuffle
    random.shuffle(image_files)

    # Calculate splits
    n = len(image_files)
    train_end = int(n * train_split)
    val_end = train_end + int(n * val_split)

    splits_data = {
        "train": image_files[:train_end],
        "val": image_files[train_end:val_end],
        "test": image_files[val_end:],
    }

    print(f"\n📊 Dataset Split:")
    print(f"   Train: {len(splits_data['train'])} images ({train_split * 100:.0f}%)")
    print(f"   Val:   {len(splits_data['val'])} images ({val_split * 100:.0f}%)")
    print(f"   Test:  {len(splits_data['test'])} images ({test_split * 100:.0f}%)")

    # Copy files
    for split_name, files in splits_data.items():
        print(f"\n📁 Copying {split_name} files...")

        for img_file in files:
            # Copy image
            dst_img = output / "images" / split_name / img_file.name
            shutil.copy2(img_file, dst_img)

            # Copy label
            label_file = source_labels / f"{img_file.stem}.txt"
            dst_label = output / "labels" / split_name / f"{img_file.stem}.txt"
            shutil.copy2(label_file, dst_label)

    # Create data.yaml
    yaml_content = f"""# Custom Dataset Configuration
# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

# Paths (relative to this file)
path: {output.absolute()}
train: images/train
val: images/val
test: images/test

# Classes
names:
"""

    for i, name in enumerate(class_names):
        yaml_content += f"  {i}: {name}\n"

    yaml_content += f"\n# Number of classes\nnc: {len(class_names)}\n"

    yaml_path = output / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\n✅ Dataset organized successfully!")
    print(f"📄 Configuration saved to: {yaml_path}")

    return yaml_path


def validate_dataset(data_yaml):
    """Validate dataset before training"""
    print("\n" + "=" * 60)
    print("🔍 VALIDATING DATASET")
    print("=" * 60)

    import yaml

    # Load config
    with open(data_yaml) as f:
        config = yaml.safe_load(f)

    dataset_path = Path(config["path"])
    issues = []

    # Check if paths exist
    for split in ["train", "val"]:
        img_dir = dataset_path / config[split]
        label_dir = dataset_path / config[split].replace("images", "labels")

        if not img_dir.exists():
            issues.append(f"Missing directory: {img_dir}")
        if not label_dir.exists():
            issues.append(f"Missing directory: {label_dir}")

    # Check for empty labels
    for split in ["train", "val"]:
        label_dir = dataset_path / config[split].replace("images", "labels")
        if label_dir.exists():
            for label_file in label_dir.glob("*.txt"):
                if label_file.stat().st_size == 0:
                    issues.append(f"Empty label: {label_file.name}")

    # Report
    if issues:
        print("⚠️  Found issues:")
        for issue in issues[:10]:
            print(f"   - {issue}")
        if len(issues) > 10:
            print(f"   ... and {len(issues) - 10} more")
        return False
    else:
        print("✅ Dataset validation passed!")

        # Show statistics
        train_images = list((dataset_path / config["train"]).glob("*.jpg"))
        train_images += list((dataset_path / config["train"]).glob("*.png"))
        val_images = list((dataset_path / config["val"]).glob("*.jpg"))
        val_images += list((dataset_path / config["val"]).glob("*.png"))

        print(f"\n📊 Dataset Statistics:")
        print(f"   Classes: {config['nc']} ({', '.join(config['names'].values())})")
        print(f"   Training images: {len(train_images)}")
        print(f"   Validation images: {len(val_images)}")

        return True


def train_model(
    data_yaml,
    model_size="n",
    epochs=100,
    batch_size=16,
    image_size=640,
    device=None,
    project="runs/train",
    name="custom_model",
    resume=False,
):
    """
    Train custom YOLO model

    Args:
        data_yaml: Path to data.yaml
        model_size: Model size (n, s, m, l, x)
        epochs: Number of epochs
        batch_size: Batch size
        image_size: Input image size
        device: Device (cuda, cpu, mps, or None for auto)
        project: Project directory
        name: Experiment name
        resume: Resume from last checkpoint
    """
    print("\n" + "=" * 60)
    print("🚀 TRAINING CUSTOM MODEL")
    print("=" * 60)

    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
            print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            print("✅ Using Apple Metal (MPS)")
        else:
            device = "cpu"
            print("⚠️  No GPU found, using CPU (will be slower)")

    print(f"\n⚙️  Training Configuration:")
    print(f"   Model: YOLOv8{model_size}")
    print(f"   Dataset: {data_yaml}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Image size: {image_size}")
    print(f"   Device: {device}")

    # Load model
    model_name = f"yolov8{model_size}.pt"
    print(f"\n📥 Loading pre-trained model: {model_name}")
    model = YOLO(model_name)

    # Train
    print(f"\n🏋️  Starting training (this may take a while)...")
    print("💡 Tip: Monitor progress in the terminal")

    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=image_size,
            device=device,
            project=project,
            name=name,
            # Optimization
            optimizer="AdamW",
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            # Augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            # Validation
            val=True,
            plots=True,
            save=True,
            save_period=10,
            # Performance
            patience=50,
            workers=8,
            cache=False,
            # Resume
            resume=resume,
            verbose=True,
        )

        print("\n✅ Training completed successfully!")

        # Show results
        output_dir = Path(project) / name
        print(f"\n📁 Results saved to: {output_dir}")
        print(f"   - Best model: {output_dir / 'weights' / 'best.pt'}")
        print(f"   - Last model: {output_dir / 'weights' / 'last.pt'}")
        print(f"   - Training plots: {output_dir / 'results.png'}")

        return results, output_dir / "weights" / "best.pt"

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def evaluate_model(model_path, data_yaml):
    """Evaluate trained model"""
    print("\n" + "=" * 60)
    print("📊 EVALUATING MODEL")
    print("=" * 60)

    model = YOLO(model_path)

    print("🔍 Running validation...")
    metrics = model.val(data=data_yaml)

    print("\n📈 Performance Metrics:")
    print(f"   mAP50:     {metrics.box.map50:.4f}")
    print(f"   mAP50-95:  {metrics.box.map:.4f}")
    print(f"   Precision: {metrics.box.mp:.4f}")
    print(f"   Recall:    {metrics.box.mr:.4f}")

    # Per-class metrics
    print("\n📊 Per-Class Performance:")
    for class_id, class_name in model.names.items():
        # Note: Simplified - actual implementation may vary
        print(f"   {class_name}: mAP50={metrics.box.map50:.4f}")

    return metrics


def test_inference(model_path, test_image_path, output_dir="test_results"):
    """Test model inference on sample images"""
    print("\n" + "=" * 60)
    print("🧪 TESTING INFERENCE")
    print("=" * 60)

    model = YOLO(model_path)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    test_path = Path(test_image_path)

    if test_path.is_dir():
        # Test on directory
        image_files = list(test_path.glob("*.jpg")) + list(test_path.glob("*.png"))
        print(f"📁 Testing on {len(image_files)} images...")

        for img_file in image_files[:5]:  # Test first 5
            results = model(str(img_file), conf=0.5)

            # Save annotated image
            for result in results:
                annotated = result.plot()
                output_file = output_path / img_file.name
                cv2.imwrite(str(output_file), annotated)

                # Print detections
                num_detections = len(result.boxes)
                print(f"   {img_file.name}: {num_detections} objects detected")
    else:
        # Test on single image
        print(f"🖼️  Testing on: {test_path.name}")
        results = model(str(test_path), conf=0.5)

        for result in results:
            annotated = result.plot()
            output_file = output_path / f"result_{test_path.name}"
            cv2.imwrite(str(output_file), annotated)

            print(f"\n✅ Detections:")
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                print(f"   - {class_name}: {confidence:.2f}")

    print(f"\n📁 Results saved to: {output_path}")


def export_model(model_path, formats=["onnx"]):
    """Export model to different formats"""
    print("\n" + "=" * 60)
    print("📦 EXPORTING MODEL")
    print("=" * 60)

    model = YOLO(model_path)

    for fmt in formats:
        print(f"\n📤 Exporting to {fmt.upper()}...")
        try:
            model.export(format=fmt)
            print(f"✅ {fmt.upper()} export successful")
        except Exception as e:
            print(f"❌ {fmt.upper()} export failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO Custom Training Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with prepared dataset
    python train_custom_yolo_example.py --data dataset/data.yaml

    # Full pipeline from raw images
    python train_custom_yolo_example.py --mode full \\
        --images dataset/raw_images \\
        --labels dataset/raw_labels \\
        --classes "cat,dog,bird"

    # Quick test (10 epochs)
    python train_custom_yolo_example.py --data dataset/data.yaml --epochs 10

    # Large model, GPU training
    python train_custom_yolo_example.py --data dataset/data.yaml \\
        --model m --epochs 300 --device cuda
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["train", "full"],
        default="train",
        help="Mode: train (use existing data.yaml) or full (organize dataset first)",
    )

    parser.add_argument("--data", type=str, help="Path to data.yaml")
    parser.add_argument(
        "--images", type=str, help="Raw images directory (for full mode)"
    )
    parser.add_argument(
        "--labels", type=str, help="Raw labels directory (for full mode)"
    )
    parser.add_argument(
        "--classes", type=str, help="Comma-separated class names (for full mode)"
    )

    parser.add_argument(
        "--model",
        choices=["n", "s", "m", "l", "x"],
        default="n",
        help="Model size (n=nano, s=small, m=medium, l=large, x=xlarge)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument(
        "--device", type=str, default=None, help="Device (cuda, cpu, mps)"
    )

    parser.add_argument("--test", type=str, help="Test images path after training")
    parser.add_argument("--export", action="store_true", help="Export model to ONNX")

    args = parser.parse_args()

    print("=" * 60)
    print("🎯 YOLO CUSTOM TRAINING")
    print("=" * 60)

    # Full pipeline
    if args.mode == "full":
        if not all([args.images, args.labels, args.classes]):
            print("❌ Full mode requires --images, --labels, and --classes")
            sys.exit(1)

        class_names = [c.strip() for c in args.classes.split(",")]

        # Organize dataset
        data_yaml = organize_dataset(
            source_images_dir=args.images,
            source_labels_dir=args.labels,
            output_dir="dataset/organized",
            class_names=class_names,
        )
    else:
        # Use existing data.yaml
        if not args.data:
            print("❌ Train mode requires --data argument")
            sys.exit(1)
        data_yaml = args.data

    # Validate dataset
    if not validate_dataset(data_yaml):
        print("\n⚠️  Dataset validation failed. Please fix issues before training.")
        sys.exit(1)

    # Train model
    results, model_path = train_model(
        data_yaml=data_yaml,
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        image_size=args.imgsz,
        device=args.device,
    )

    if model_path is None:
        print("\n❌ Training failed!")
        sys.exit(1)

    # Evaluate
    evaluate_model(model_path, data_yaml)

    # Test inference
    if args.test:
        test_inference(model_path, args.test)

    # Export
    if args.export:
        export_model(model_path, formats=["onnx"])

    print("\n" + "=" * 60)
    print("✅ CUSTOM TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n🎯 Your trained model: {model_path}")
    print(f"\n📚 Next steps:")
    print("   1. Test on more images")
    print("   2. Fine-tune if needed")
    print("   3. Deploy to production")
    print("   4. See wiki/YOLO-Custom-Training.md for details")


if __name__ == "__main__":
    main()
