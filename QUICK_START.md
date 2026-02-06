# Quick Start Guide

## ✅ Application is Ready!

### Run Detection App
```bash
py -3.11 -m streamlit run app.py
```

**Features:**
- Detects all 80 COCO classes (person, car, dog, cat, etc.)
- Image upload mode
- Live camera mode (auto-detects resolution)
- Adjustable confidence threshold

---

### Run Training Data Manager
```bash
py -3.11 -m streamlit run train_manager.py
```

**Features:**
- Add your custom class names
- Upload training images via drag & drop
- Auto-organize into train/val/test folders
- Generate data.yaml automatically
- View dataset statistics

---

## Simple Training Workflow

### 1. Prepare Data (train_manager.py)
```bash
py -3.11 -m streamlit run train_manager.py
```
- Add class names (Dog, Cat, Car, etc.)
- Upload images
- Click "Generate data.yaml"

### 2. Label Images

**Option A - Roboflow.com (Easiest):**
1. Go to roboflow.com
2. Create free account
3. Upload images from `dataset/images/train/`
4. Draw boxes and assign labels
5. Export as "YOLOv8" format
6. Download and extract labels to `dataset/labels/train/`

**Option B - LabelImg:**
```bash
pip install labelImg
labelImg
```

### 3. Train Model
```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=50
```

### 4. Use Your Model
```bash
# Copy trained model
copy runs\detect\train\weights\best.pt models\my_model.pt
```

**Edit config.py:**
```python
MODEL_PATH = "models/my_model.pt"
```

**Run app:**
```bash
py -3.11 -m streamlit run app.py
```

---

## Files Overview

| File | Purpose |
|------|---------|
| `app.py` | Main object detection application |
| `train_manager.py` | Training data management tool |
| `config.py` | Configuration settings |
| `utils/detector.py` | YOLOv8 detection engine |
| `utils/camera.py` | Camera handling (auto-detect) |
| `utils/visualizer.py` | Drawing boxes and labels |
| `data.yaml` | Training dataset config |
| `requirements.txt` | Python dependencies |

---

## Camera Troubleshooting

If camera doesn't work:
1. See `CAMERA_FIX.md`
2. Run: `py -3.11 test_camera.py`
3. Enable "Let desktop apps access camera" in Windows Settings

---

## Current Status

✅ **Working:**
- Standard YOLOv8n with 80 COCO classes
- Image upload detection
- Live camera with auto-detect resolution
- Training data manager tool
- Flexible custom training support

✅ **Fixed:**
- Removed hardcoded camera resolution
- Removed hardcoded FPS settings
- Removed custom 5-class restriction
- Uses native YOLO class names

---

**Ready to use!** Choose what you want to do:
- Detect objects → `py -3.11 -m streamlit run app.py`
- Prepare training data → `py -3.11 -m streamlit run train_manager.py`
