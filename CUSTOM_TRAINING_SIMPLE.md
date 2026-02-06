# Custom Training Guide - Train Your Own Model

## WHY TRAIN A CUSTOM MODEL?

The current app uses YOLOv8 pretrained on COCO dataset, which:
- Doesn't know Gold artifacts, Stone, or Soil
- Identifies persons correctly now (FIXED!)
- May confuse objects

**With YOUR custom trained model:**
- Detects EXACTLY your materials (Gold, Plastic, Stone, Soil, Humans)
- Much more accurate for archaeological excavation
- Recognizes rare earth materials if you label them

---

## HOW TO TRAIN (SIMPLE STEPS)

### 1. Collect Images (50-200 per class)

Take photos of:
- Gold artifacts
- Plastic contamination  
- Stone (rocks, carved artifacts)
- Soil (different types)
- Humans (for safety detection)

### 2. Label Your Images

**EASIEST METHOD - Use Roboflow:**

1. Go to https://roboflow.com
2. Create free account
3. Click "Create New Project"
4. Upload all your images
5. Draw boxes around each object
6. Label as: Gold, Plastic, Stone, Soil, or Humans
7. Export as "YOLOv8" format
8. Download ZIP
9. Extract to `project1/dataset/` folder

**Example:**
```
You upload a photo of excavation site
Draw box around gold coin -> Label: "Gold"
Draw box around plastic bottle -> Label: "Plastic"  
Draw box around person -> Label: "Humans"
Save and repeat for all images
```

### 3. Train Your Model

```bash
# Quick training (25 epochs, ~1-2 hours on CPU)
yolo detect train data=data.yaml model=yolov8n.pt epochs=25 imgsz=640 batch=8

# Full training (50 epochs, better accuracy)
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

### 4. Use Your Trained Model

```bash
# Copy trained model
copy runs\detect\train\weights\best.pt models\my_model.pt

# Edit config.py, change line 7 to:
MODEL_PATH = "models/my_model.pt"

# Run app
py -3.11 -m streamlit run app.py
```

NOW it will detect YOUR materials accurately!

---

## LABEL FORMAT

Each image needs a .txt file with same name:

**image001.jpg** -> **image001.txt** containing:
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.15
```

Format: `<class_id> <x_center> <y_center> <width> <height>`

Class IDs:
- 0 = Gold
- 1 = Plastic
- 2 = Stone
- 3 = Soil
- 4 = Humans

All values are normalized 0-1 (0.5 = center of image)

---

## FOLDER STRUCTURE

```
project1/
└── dataset/
    ├── images/
    │   ├── train/      <- 70% of your images
    │   ├── val/        <- 20% of your images
    │   └── test/       <- 10% of your images
    └── labels/
        ├── train/      <- Matching .txt files
        ├── val/
        └── test/
```

---

## TRAINING TIPS

1. **More data = Better accuracy**
   - 50 images minimum per class
   - 200+ images ideal

2. **Vary your photos**
   - Different lighting
   - Different angles
   - Different backgrounds
   - Close-up and far away

3. **Label carefully**
   - Draw tight boxes (not too big)
   - Don't miss any objects
   - Use correct class names

4. **Monitor training**
   - Training will show you loss graphs
   - Lower loss = better model
   - Watch for "best.pt" being saved

---

## AFTER TRAINING

Your trained model will be at:
`runs/detect/train/weights/best.pt`

Metrics to check:
- mAP (mean Average Precision) - Higher is better (aim for >0.5)
- Precision - Fewer false positives
- Recall - Fewer missed detections

Test your model:
```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=test_image.jpg
```

---

## QUICK COMMANDS REFERENCE

```bash
# 1. Install labeling tool (optional)
pip install labelImg
labelImg

# 2. Train model (CPU - slow but works)
yolo detect train data=data.yaml model=yolov8n.pt epochs=25 imgsz=640 batch=4

# 3. Train model (GPU - fast)
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640 batch=16

# 4. Test model
yolo detect predict model=runs/detect/train/weights/best.pt source=dataset/images/test/

# 5. Use in app
# Update config.py line 7: MODEL_PATH = "models/best.pt"
py -3.11 -m streamlit run app.py
```

---

## NEED HELP?

See full guide: TRAINING_GUIDE.md
Or ask me for specific help!
