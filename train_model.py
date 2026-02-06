"""
YOLOv8 Training Script
Can be run directly or called by train_manager.py
"""
import argparse
from ultralytics import YOLO
import torch
import sys

def train(epochs=50, imgsz=640, batch=16, model_name='yolov8n.pt', device=None):
    print("=" * 70)
    print("YOLOv8 Training Launcher")
    print("=" * 70)
    
    # Auto-detect device if not specified
    if device is None:
        device = '0' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Image Size: {imgsz}")
    print(f"Batch Size: {batch}")
    print("-" * 70)

    try:
        # Load model
        model = YOLO(model_name)
        
        # Train
        results = model.train(
            data='data.yaml',
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            patience=10,
            save=True,
            project='runs/detect',
            name='custom_model',
            exist_ok=True  # Overwrite existing experiment
        )
        
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETE!")
        print("=" * 70)
        print("Best model: runs/detect/custom_model/weights/best.pt")
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    args = parser.parse_args()
    
    train(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, model_name=args.model)
