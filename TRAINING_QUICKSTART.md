# Quick Start: Train Your Archaeological Detection Model

## ✅ Setup Complete

Your training infrastructure is ready:
- ✓ `data.yaml` - Dataset configuration (5 classes)
- ✓ `dataset/` - Organized directory structure
- ✓ `train_model.py` - Training script
- ✓ `TRAINING_GUIDE.md` - Comprehensive guide

## 📋 Next Steps

### 1. Prepare Your Dataset

**You need labeled images of:**
- 🧱 Brick (clay bricks, ceramics)
- 🪨 Stone (flint, carved stone)
- 🔵 Plastic (contamination)
- 🟡 Metal (coins, tools, artifacts)
- 💍 Ring (circular artifacts)

**Recommended: 100-500 images per class**

### 2. Label Your Images

**Option A - Roboflow (Easy):**
1. Visit https://roboflow.com
2. Upload images
3. Draw bounding boxes
4. Label as: Brick, Stone, Plastic, Metal, Ring
5. Export as "YOLOv8 format"
6. Extract to `dataset/` folder

**Option B - LabelImg:**
```bash
pip install labelimg
labelimg
```

### 3. Organize Dataset

Place images and labels in:
```
dataset/
├── images/
│   ├── train/     ← 70-80% of images
│   ├── val/       ← 10-20% of images
│   └── test/      ← 10% of images
└── labels/
    ├── train/     ← Corresponding .txt files
    ├── val/
    └── test/
```

### 4. Start Training

**Option 1 - Using the script:**
```bash
py -3.11 train_model.py
```

**Option 2 - Your command:**
```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

**Option 3 - Custom Python:**
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='data.yaml', epochs=50, imgsz=640, batch=16)
```

### 5. After Training

```bash
# Copy trained model
copy runs\detect\train\weights\best.pt models\archaeological_model.pt

# Update config.py
# Change: MODEL_PATH = "models/archaeological_model.pt"

# Test the app
py -3.11 -m streamlit run app.py
```

## ⏱️ Expected Training Time

- **CPU**: 4-8 hours (500 images, 50 epochs)
- **GPU**: 30-60 minutes (500 images, 50 epochs)

## 🎯 Training Tips

1. **Start small**: Try 25 epochs first to verify setup
2. **Monitor progress**: Watch loss curves in `runs/detect/train/`
3. **GPU memory issues**: Reduce batch size (`batch=8` or `batch=4`)
4. **Poor results**: Collect more data or increase epochs

## 📚 Full Documentation

See `TRAINING_GUIDE.md` for:
- Detailed labeling instructions
- Troubleshooting guide
- Performance optimization
- Dataset requirements

## 🆘 Quick Troubleshooting

**"No images found"**: Check that images are in `dataset/images/train/`
**Out of memory**: Reduce batch size to 8 or 4
**Training too slow**: Use GPU or reduce epochs to 25

---

**Ready to train? Follow steps 1-4 above!**
