"""
Simple Training Data Manager
Add images and create labels easily for YOLOv8 training
"""
import os
import streamlit as st
from pathlib import Path
import shutil
from PIL import Image
import cv2
import numpy as np
import sys

st.set_page_config(page_title="YOLOv8 Training Data Manager", page_icon="📦", layout="wide")

# Initialize session state
if 'class_names' not in st.session_state:
    st.session_state.class_names = []
if 'current_class' not in st.session_state:
    st.session_state.current_class = None

# Create dataset directories
DATASET_DIR = Path("dataset")
for split in ['train', 'val', 'test']:
    (DATASET_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

st.title("📦 YOLOv8 Training Data Manager")
st.markdown("Simple tool to organize training images and create data.yaml")

# Sidebar - Class Management
with st.sidebar:
    st.header("📋 Class Management")
    
    # Add new class
    new_class = st.text_input("Add New Class")
    if st.button("Add Class") and new_class:
        if new_class not in st.session_state.class_names:
            st.session_state.class_names.append(new_class)
            st.success(f"Added: {new_class}")
        else:
            st.warning("Class already exists")
    
    # Display current classes
    if st.session_state.class_names:
        st.subheader("Current Classes:")
        for idx, cls in enumerate(st.session_state.class_names):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"{idx}: {cls}")
            with col2:
                if st.button("❌", key=f"del_{idx}"):
                    st.session_state.class_names.pop(idx)
                    st.rerun()
    else:
        st.info("No classes added yet")
    
    st.divider()
    
    # Generate data.yaml
    if st.button("📄 Generate data.yaml", type="primary"):
        if st.session_state.class_names:
            yaml_content = f"""# YOLOv8 Dataset Configuration
path: {DATASET_DIR.absolute()}
train: images/train
val: images/val
test: images/test

nc: {len(st.session_state.class_names)}

names:
"""
            for idx, cls in enumerate(st.session_state.class_names):
                yaml_content += f"  {idx}: {cls}\n"
            
            with open("data.yaml", "w") as f:
                f.write(yaml_content)
            
            st.success("✅ data.yaml created!")
            st.code(yaml_content, language="yaml")
        else:
            st.error("Add classes first!")

# Main area - Image Management
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Auto-Label", "📁 View Dataset", "🚀 Train", "🌍 Web Datasets"])

with tab4:
    st.header("🌍 Ready-to-Use Public Datasets")
    st.markdown("Don't have images? Download ready-made datasets from the web!")
    
    st.info("💡 **Tip:** Download in **YOLOv8 format**, extract to `dataset/` folder, and you're ready to train!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Roboflow Universe")
        st.write("Best source for YOLOv8 datasets.")
        
        st.markdown("""
        - [🔍 Search "Archaeology"](https://universe.roboflow.com/search?q=archaeology)
        - [🔍 Search "Stone Tools"](https://universe.roboflow.com/search?q=stone%20tools)
        - [🔍 Search "Pottery"](https://universe.roboflow.com/search?q=pottery)
        - [🔍 Search "Coins"](https://universe.roboflow.com/search?q=coins)
        """)
        
        st.markdown("### How to install:")
        st.code("""
# 1. Download ZIP from Roboflow
# 2. Extract contents
# 3. Copy images to dataset/images/train/
# 4. Copy labels to dataset/labels/train/
        """)
        
    with col2:
        st.subheader("📊 Kaggle & Others")
        st.markdown("""
        - [Search Kaggle](https://www.kaggle.com/search?q=archaeology+object+detection)
        - [Zenodo (Academic)](https://zenodo.org/search?q=archaeology%20dataset)
        """)
        
        st.success("""
        **Recommended Strategy:**
        1. Download a "Coin" dataset for **Gold** class
        2. Download "Stone tool" dataset for **Stone** class
        3. Combine them into your local dataset
        4. Train a powerful mixed model!
        """)

with tab1:
    st.header("Upload & Auto-Label Images")
    st.markdown("Upload images and automatically assign them to a class.")
    
    col1, col2 = st.columns(2)
    with col1:
        # Select split
        split = st.selectbox("Dataset Split", ['train', 'val', 'test'], 
                             help="train=70-80%, val=10-20%, test=10%")
    
    with col2:
        # Select class for auto-labeling
        if st.session_state.class_names:
            target_class_Name = st.selectbox("Select Class for these images:", st.session_state.class_names)
            target_class_id = st.session_state.class_names.index(target_class_Name)
            auto_label = st.checkbox("Auto-label objects?", value=True, 
                                   help="If checked, creates a box around the whole image for the selected class.")
        else:
            st.warning("⚠️ Add classes in the sidebar first!")
            target_class_Name = None
            auto_label = False

    # Upload images
    uploaded_files = st.file_uploader(
        "Upload Images", 
        type=['jpg', 'jpeg', 'png'], 
        accept_multiple_files=True
    )
    
    if uploaded_files and target_class_Name:
        st.write(f"Ready to save {len(uploaded_files)} images as **{target_class_Name}**")
        
        if st.button("Save & Label Images"):
            saved_count = 0
            for file in uploaded_files:
                # Save image
                img_path = DATASET_DIR / 'images' / split / file.name
                # Handle duplicate names
                if img_path.exists():
                    name_base = Path(file.name).stem
                    import uuid
                    new_name = f"{name_base}_{str(uuid.uuid4())[:8]}{Path(file.name).suffix}"
                    img_path = DATASET_DIR / 'images' / split / new_name
                
                # Write image file
                with open(img_path, "wb") as f:
                    f.write(file.getbuffer())
                
                # Create label file
                label_path = DATASET_DIR / 'labels' / split / f"{img_path.stem}.txt"
                
                if auto_label:
                    # Create a bounding box covering most of the image
                    # Format: class_id x_center y_center width height
                    # Using 0.5 0.5 0.9 0.9 to cover 90% of image with safe margin
                    with open(label_path, "w") as f:
                        f.write(f"{target_class_id} 0.5 0.5 0.95 0.95")
                else:
                    # Create empty file (background/negative sample)
                    label_path.touch()
                
                saved_count += 1
            
            st.success(f"✅ Saved & Labeled {saved_count} images as '{target_class_Name}' in {split}/")
            if auto_label:
                st.info("Labels created automatically! (Assuming one object per image)")
            else:
                st.info("Uploaded without labels. Use external tool to draw boxes.")
    elif uploaded_files and not target_class_Name:
        st.error("Please add and select a class first!")

with tab2:
    st.header("Dataset Overview")
    
    # Count images in each split
    for split in ['train', 'val', 'test']:
        img_dir = DATASET_DIR / 'images' / split
        label_dir = DATASET_DIR / 'labels' / split
        
        img_count = len(list(img_dir.glob('*')))
        label_count = len([f for f in label_dir.glob('*.txt') if f.stat().st_size > 0])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{split.upper()} Images", img_count)
        with col2:
            st.metric(f"{split.upper()} Labeled", label_count)
        with col3:
            coverage = (label_count / img_count * 100) if img_count > 0 else 0
            st.metric("Coverage", f"{coverage:.0f}%")
    
    st.divider()
    
    # Show sample images
    st.subheader("Sample Images")
    split_view = st.selectbox("View Split", ['train', 'val', 'test'], key='view')
    img_dir = DATASET_DIR / 'images' / split_view
    images = list(img_dir.glob('*'))[:6]
    
    if images:
        cols = st.columns(3)
        for idx, img_path in enumerate(images):
            with cols[idx % 3]:
                img = Image.open(img_path)
                st.image(img, caption=img_path.name, use_column_width=True)
    else:
        st.info(f"No images in {split_view}/ yet")

with tab3:
    st.header("🚀 Start Training")
    
    # Check if data.yaml exists
    if not Path("data.yaml").exists():
        st.error("❌ Generate data.yaml first (see sidebar)")
    else:
        st.success("✅ data.yaml found")
        
        # Training parameters
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.number_input("Epochs", min_value=1, max_value=500, value=50)
            batch = st.number_input("Batch Size", min_value=1, max_value=32, value=8)
        with col2:
            imgsz = st.selectbox("Image Size", [416, 640, 1280], index=1)
            model = st.selectbox("Model", ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt'])
        
        # Generate command
        train_cmd = f"yolo detect train data=data.yaml model={model} epochs={epochs} imgsz={imgsz} batch={batch}"
        
        st.code(train_cmd, language="bash")
        
        st.info("""
        **To start training:**
        1. Copy the command above
        2. Open PowerShell/Terminal
        3. Paste and run the command
        4. Wait for training to complete
        5. Find your model at: `runs/detect/train/weights/best.pt`
        """)
        
        # Run button
        if st.button("🚀 RUN TRAINING NOW", type="primary"):
            st.warning("⚠️ Training started! Check your terminal console for progress bars.")
            
            # Create command list
            cmd = [
                sys.executable, "train_model.py",
                "--epochs", str(epochs),
                "--imgsz", str(imgsz),
                "--batch", str(batch),
                "--model", model
            ]
            
            # Show command being run
            st.info(f"Executing: {' '.join(cmd)}")
            
            # Run in a container for real-time output
            output_container = st.empty()
            
            try:
                import subprocess
                
                # Run process
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Stream output
                output_log = ""
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        # Print to console
                        print(output.strip())
                        # Update UI (limit length to avoid lagging)
                        output_log += output
                        if len(output_log) > 2000:
                            output_log = output_log[-2000:]
                        output_container.code(output_log)
                
                rc = process.poll()
                
                if rc == 0:
                    st.success("✅ Training completed successfully!")
                    st.balloons()
                else:
                    st.error(f"❌ Training failed with exit code {rc}")
                    
            except Exception as e:
                st.error(f"Failed to run training: {e}")
        
        st.divider()
        st.markdown("### After Training:")
        st.code("""
# Copy model to models folder
copy runs\\detect\\custom_model\\weights\\best.pt models\\my_model.pt

# Update config.py
MODEL_PATH = "models/my_model.pt"

# Run detection app
py -3.11 -m streamlit run app.py
        """, language="bash")

# Instructions
with st.expander("📖 How to Use This Tool"):
    st.markdown("""
    ### Step-by-Step Guide:
    
    1. **Add Classes** (Sidebar)
       - Enter class name (e.g. "Cat", "Gold") and click Add
    
    2. **Upload & Auto-Label**
       - Select the class (e.g. "Cat")
       - Check "Auto-label objects?"
       - Upload all your Cat images
       - Click "Save & Label"
       - (Repeat for other classes)
    
    3. **Generate Config**
       - Click "Generate data.yaml" in sidebar
    
    4. **Train**
       - Go to "Train" tab
       - Copy command and run!
    
    ### Note on Auto-Labeling:
    This tool assumes **one object per image** filling most of the frame.
    For complex images with multiple objects, uncheck "Auto-label" and use Roboflow/LabelImg.
    
    4. **Generate data.yaml**
       - Click "Generate data.yaml" in sidebar
       - File will be created automatically
    
    5. **Train Model**
       - Go to "Train" tab
       - Copy the training command
       - Run in terminal
       - Wait for completion
    
    6. **Use Your Model**
       - Copy `best.pt` to models folder
       - Update `config.py`
       - Run main app!
    
    ### Recommended Dataset Split:
    - **Train**: 70-80% of images
    - **Val**: 10-20% of images
    - **Test**: 10% of images
    
    ### Tips:
    - Aim for 50-200 images per class
    - Use varied lighting and angles
    - Label all objects in each image
    - More data = better accuracy
    """)

st.divider()
st.caption("💡 Tip: For best results, collect diverse images and label carefully!")
