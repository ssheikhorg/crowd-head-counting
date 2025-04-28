import os
from ultralytics import YOLO
import argparse
from pathlib import Path
import torch


def train_model(data_yaml, epochs=100, imgsz=640, batch=16, resume=False):
    # Verify paths
    data_path = Path(data_yaml)
    if not data_path.exists():
        raise FileNotFoundError(f"YAML not found at {data_path}")

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Determine model source
    model_path = "yolov8m.pt"  # Default pretrained model
    if resume:
        resume_path = Path("models/last.pt")
        if resume_path.exists():
            model_path = str(resume_path)
            print(f"Resuming training from {model_path}")
        else:
            print("Warning: No checkpoint found, starting from pretrained weights")

    # Load model
    model = YOLO(model_path).to(device)

    # Training args
    args = {
        "data": str(data_path),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "optimizer": "AdamW",
        "lr0": 0.01,
        "cache": "ram",
        "workers": min(4, os.cpu_count()),
        "single_cls": True,
        "project": "runs_optimized",
        "name": "yolov8m_crowd",
        "exist_ok": True,
        "resume": resume,  # Critical for proper resumption
    }

    # Train
    results = model.train(**args)

    # Export
    model.export(format="onnx")
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Training with Resume Support")
    parser.add_argument(
        "--data", default="data/crowd.yaml", help="Path to data YAML file"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs")
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Image size for training"
    )
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from last checkpoint"
    )
    args = parser.parse_args()

    train_model(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        resume=args.resume,
    )
