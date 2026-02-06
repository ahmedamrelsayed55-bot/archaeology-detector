"""
Easy Custom Training Assistant
Upload your own labeled images to train a custom YOLOv8 model
"""
import os
import shutil
from pathlib import Path

print("""
================================================================================
    YOLOv8 CUSTOM TRAINING ASSISTANT
    Train your own archaeological material detector
================================================================================

This tool helps you organize your labeled images for training.

STEP 1: Prepare Your Images
----------------------------
You need:
- Images of Gold, Plastic, Stone, Soil, and Humans
- At least 50-100 images per class (more is better)
- Clear, well-lit photos from excavation sites

STEP 2: Label Your Images
----------------------------
Option A - Use Roboflow (EASIEST):
  1. Go to: https://roboflow.com
  2. Create free account
  3. Upload all your images
  4. Draw bounding boxes around objects
  5. Label each box as: Gold, Plastic, Stone, Soil, or Humans
  6. Export in "YOLOv8" format
  7. Download the ZIP file
  8. Extract to: project1/dataset/

Option B - Use LabelImg:
  1. Install: pip install labelImg
  2. Run: labelImg
  3. Open your image folder
  4. Press 'w' to draw box
  5. Select class name
  6. Press 's' to save
  7. Converts to YOLO format

STEP 3: Organize Dataset
----------------------------
Your dataset folder should look like this:

dataset/
├── images/
│   ├── train/       <- Put 70-80% of images here
│   ├── val/         <- Put 10-20% of images here
│   └── test/        <- Put 10% of images here
└── labels/
    ├── train/       <- Matching .txt label files
    ├── val/
    └── test/

Each image needs a matching .txt file with same name:
- img001.jpg  →  img001.txt
- img002.jpg  →  img002.txt

Label format (each line):
<class_id> <x_center> <y_center> <width> <height>

Class IDs:
  0 = Gold
  1 = Plastic
  2 = Stone
  3 = Soil
  4 = Humans

Example label file (img001.txt):
0 0.5 0.5 0.3 0.4
4 0.2 0.8 0.1 0.2

STEP 4: Train Your Model
----------------------------
Once dataset is ready, run ONE of these commands:

Option A - Simple command:
  yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640

Option B - Using Python script:
  py -3.11 train_model.py

Option C - Quick test (25 epochs):
  yolo detect train data=data.yaml model=yolov8n.pt epochs=25 imgsz=640 batch=8

Training will take:
- CPU: 2-8 hours (depending on dataset size)
- GPU: 30 minutes - 2 hours

STEP 5: Use Your Trained Model
----------------------------
After training finishes:

1. Find your model:
   runs/detect/train/weights/best.pt

2. Copy to models folder:
   copy runs\\detect\\train\\weights\\best.pt models\\custom_model.pt

3. Update config.py:
   MODEL_PATH = "models/custom_model.pt"

4. Run the app:
   py -3.11 -m streamlit run app.py

Your custom model will now detect YOUR specific materials!

================================================================================

QUICK START CHECKLIST:
□ Collect 50-100 images per class
□ Label images (Roboflow or LabelImg)
□ Organize into dataset/images/train, val, test
□ Ensure labels match images
□ Run training command
□ Copy best.pt to models/
□ Update config.py
□ Test in app!

TIPS FOR BETTER ACCURACY:
- More data = better results (aim for 200+ images per class)
- Use varied lighting conditions
- Include different angles
- Use clear, focused images
- Label carefully and consistently

Need help? See TRAINING_GUIDE.md for detailed instructions.

================================================================================
""")

# Check if dataset structure exists
dataset_path = Path("dataset")
if dataset_path.exists():
    print("\n[OK] Dataset folder found!")
    
    # Check structure
    required = [
        "images/train",
        "images/val", 
        "images/test",
        "labels/train",
        "labels/val",
        "labels/test"
    ]
    
    for folder in required:
        path = dataset_path / folder
        if path.exists():
            count = len(list(path.glob("*")))
            print(f"  [OK] {folder}: {count} files")
        else:
            print(f"  [!] Missing: {folder}")
else:
    print("\n[!] Dataset folder not found")
    print("    Run this to create structure:")
    print("    mkdir dataset\\images\\train dataset\\images\\val dataset\\images\\test")
    print("    mkdir dataset\\labels\\train dataset\\labels\\val dataset\\labels\\test")

print("\n" + "="*80)
print("Ready to label your images and train!")
print("="*80)
