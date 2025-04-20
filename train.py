import os
from ultralytics import YOLO
import argparse
from pathlib import Path
import torch


def train_model(data_yaml, epochs=100, imgsz=640, batch=16):
    # Verify paths
    data_path = Path(data_yaml)
    if not data_path.exists():
        raise FileNotFoundError(f"YAML not found at {data_path}")

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    model = YOLO('yolov8m.pt').to(device)

    # Training args
    args = {
        'data': str(data_path),
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'device': device,
        'optimizer': 'AdamW',
        'lr0': 0.01,
        'cache': 'ram',  # Cache in memory
        'workers': min(4, os.cpu_count()),
        'single_cls': True,
        'project': 'runs_optimized',
        'name': 'yolov8m_crowd',
        'exist_ok': True,
    }

    # Train
    results = model.train(**args)

    # Export
    model.export(format="onnx")
    print("Optimized training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/crowd.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    train_model(args.data, args.epochs, args.imgsz, args.batch)