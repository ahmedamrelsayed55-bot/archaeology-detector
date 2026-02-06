# YOLOv8 Training Guide - Generic Version

## Train on YOUR Custom Data

This guide shows you how to train YOLOv8 on ANY custom dataset with ANY number of classes.

---

## Quick Start

### 1. Prepare Your Dataset

Create a `data.yaml` file defining your classes:

```yaml
path: ./dataset
train: images/train
val: images/val
test: images/test

nc: 5  # Number of classes (change as needed)

names:
  0: YourClass1
  1: YourClass2
  2: YourClass3
  3: YourClass4
  4: YourClass5
```

### 2. Organize Your Images

```
dataset/
├── images/
│   ├── train/    # 70-80% of images
│   ├── val/      # 10-20% of images
│   └── test/     # 10% of images
└── labels/
    ├── train/    # Matching .txt files
    ├── val/
    └── test/
```

### 3. Label Your Images

**Recommended: Use Roboflow.com**
1. Upload images
2. Draw bounding boxes
3. Assign class labels
4. Export as "YOLOv8" format
5. Download and extract to `dataset/`

**Label Format (.txt files):**
```
<class_id> <x_center> <y_center> <width> <height>
```
All values normalized 0-1

### 4. Train Your Model

```bash
# Basic training
yolo detect train data=data.yaml model=yolov8n.pt epochs=50

# With custom settings
yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=16

# Quick test (CPU friendly)
yolo detect train data=data.yaml model=yolov8n.pt epochs=25 batch=4
```

### 5. Use Your Trained Model

```bash
# Copy trained model
copy runs\detect\train\weights\best.pt models\my_custom_model.pt
```

**Edit config.py:**
```python
MODEL_PATH = "models/my_custom_model.pt"
```

**Run app:**
```bash
py -3.11 -m streamlit run app.py
```

---

## Training Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `epochs` | Training iterations | 50-100 |
| `imgsz` | Image size | 640 |
| `batch` | Batch size | 16 (GPU), 4-8 (CPU) |
| `patience` | Early stopping | 10 |
| `lr0` | Initial learning rate | 0.01 |

---

## Model Sizes

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| yolov8n.pt | Smallest | Fastest | Good |
| yolov8s.pt | Small | Fast | Better |
| yolov8m.pt | Medium | Moderate | Best |
| yolov8l.pt | Large | Slow | Excellent |

---

## Example: Training on 3 Custom Classes

**data.yaml:**
```yaml
path: ./my_dataset
train: images/train
val: images/val

nc: 3

names:
  0: Cat
  1: Dog
  2: Bird
```

**Train:**
```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=50
```

**Result:**
- Model at: `runs/detect/train/weights/best.pt`
- Can detect: Cat, Dog, Bird

---

## Troubleshooting

**Out of memory:**
```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=25 batch=4 imgsz=416
```

**Poor accuracy:**
- Collect more data (200+ images per class)
- Increase epochs (100+)
- Use larger model (yolov8s or yolov8m)
- Improve label quality

**Training too slow:**
- Use GPU
- Reduce batch size
- Use smaller model (yolov8n)

---

## Advanced Training

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='my_custom_detector',
    patience=20,
    save=True,
    device=0,  # GPU ID, or 'cpu'
    workers=8,
    optimizer='AdamW',
    lr0=0.01,
    plots=True
)
```

---

## Ready to Use

The application now:
- ✅ Uses standard YOLOv8n (80 COCO classes)
- ✅ Auto-detects camera capabilities
- ✅ Supports ANY custom trained model
- ✅ No hardcoded classes or resolutions

**Train your model → Update config.py → Run app!**
