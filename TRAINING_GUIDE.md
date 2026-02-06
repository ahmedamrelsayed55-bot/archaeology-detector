# YOLOv8 Training Guide for Archaeological Material Detection

## Dataset Preparation

### 1. Directory Structure

Your dataset should be organized as follows:

```
project1/
├── dataset/
│   ├── images/
│   │   ├── train/          # Training images (70-80%)
│   │   │   ├── img001.jpg
│   │   │   ├── img002.jpg
│   │   │   └── ...
│   │   ├── val/            # Validation images (10-20%)
│   │   │   ├── img101.jpg
│   │   │   └── ...
│   │   └── test/           # Test images (10%)
│   │       ├── img201.jpg
│   │       └── ...
│   └── labels/
│       ├── train/          # Training labels
│       │   ├── img001.txt
│       │   ├── img002.txt
│       │   └── ...
│       ├── val/            # Validation labels
│       │   ├── img101.txt
│       │   └── ...
│       └── test/           # Test labels
│           ├── img201.txt
│           └── ...
├── data.yaml
└── train_model.py
```

### 2. Collecting Images

**Recommended Dataset Size:**
- Minimum: 100-200 images per class (500-1000 total)
- Good: 500+ images per class (2500+ total)
- Excellent: 1000+ images per class (5000+ total)

**Image Requirements:**
- Format: JPG, PNG
- Resolution: At least 640x640 pixels
- Quality: Clear, well-lit images
- Variety: Different angles, lighting, backgrounds

**What to photograph:**
- **Brick**: Clay bricks, pottery shards, fired ceramics
- **Stone**: Flint tools, carved stone, natural rocks
- **Plastic**: Modern bottles, contamination, packaging
- **Metal**: Coins, tools, jewelry, bronze artifacts
- **Ring**: Ring-shaped items, circular artifacts

### 3. Labeling Images

**Option A: Using Roboflow (Recommended)**
1. Visit https://roboflow.com
2. Create free account
3. Upload images
4. Draw bounding boxes around objects
5. Assign class labels (Brick, Stone, Plastic, Metal, Ring)
6. Export in "YOLOv8" format
7. Download and extract to `dataset/` folder

**Option B: Using LabelImg**
```bash
pip install labelImg
labelImg
```
1. Open image directory
2. Draw bounding boxes (keyboard: 'w')
3. Select class from dropdown
4. Save (keyboard: 's')
5. Manually convert to YOLO format if needed

**Label Format (YOLO):**
Each image needs a corresponding `.txt` file with format:
```
<class_id> <x_center> <y_center> <width> <height>
```

Example `img001.txt`:
```
3 0.5 0.5 0.3 0.4
0 0.2 0.3 0.15 0.2
```
- All values are normalized (0-1)
- class_id: 0=Brick, 1=Stone, 2=Plastic, 3=Metal, 4=Ring

### 4. Dataset Split

**Recommended split:**
- Training: 70-80%
- Validation: 10-20%
- Test: 10%

## Training the Model

### Option 1: Using the Training Script

```bash
py -3.11 train_model.py
```

### Option 2: Command Line (Your Command)

```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

### Option 3: Python Script

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=16
)
```

## Training Parameters Explained

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `epochs` | Number of training cycles | 50-100 |
| `imgsz` | Image size for training | 640 |
| `batch` | Batch size (GPU memory dependent) | 8-16 (CPU), 16-32 (GPU) |
| `lr0` | Initial learning rate | 0.01 |
| `patience` | Early stopping patience | 10 |

## After Training

### 1. Locate Trained Model

After training completes, find your model at:
```
runs/detect/train/weights/best.pt
```

### 2. Copy to Models Directory

```bash
copy runs\detect\train\weights\best.pt models\best.pt
```

### 3. Update Configuration

Edit `config.py`:
```python
MODEL_PATH = "models/best.pt"  # Use your trained model
```

### 4. Test the Model

Run the application:
```bash
py -3.11 -m streamlit run app.py
```

Try both Image Upload and Live Camera modes!

## Troubleshooting

### Issue: "No module named 'ultralytics'"
```bash
pip install ultralytics
```

### Issue: Out of memory (GPU)
Reduce batch size:
```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640 batch=8
```

### Issue: Training very slow (CPU)
- Use smaller model: `yolov8n.pt`
- Reduce image size: `imgsz=416`
- Reduce batch size: `batch=4`
- Use fewer epochs: `epochs=25`

### Issue: Poor accuracy
- Collect more training data
- Improve label quality
- Increase epochs (100-200)
- Try larger model: `yolov8s.pt` or `yolov8m.pt`
- Add data augmentation

## Performance Tips

1. **Use GPU if available**: Training on GPU is 10-50x faster
2. **Start small**: Train with 25 epochs first to verify setup
3. **Monitor training**: Watch the loss curves in `runs/detect/train/`
4. **Validate regularly**: Check validation metrics during training
5. **Save checkpoints**: Training can be resumed if interrupted

## Expected Training Time

| Hardware | Dataset Size | Epochs | Approximate Time |
|----------|--------------|--------|------------------|
| CPU | 500 images | 50 | 4-8 hours |
| CPU | 2000 images | 50 | 12-24 hours |
| GPU (RTX 3060) | 500 images | 50 | 30-60 minutes |
| GPU (RTX 3060) | 2000 images | 50 | 2-4 hours |

## Next Steps

1. ✅ Prepare dataset (images + labels)
2. ✅ Organize into train/val/test folders
3. ✅ Run training command
4. ✅ Copy best.pt to models/
5. ✅ Update config.py
6. ✅ Test in application
7. ✅ Evaluate on real archaeological images
