"""
Archaeological Material Detection Application - Enhanced with Live Camera
A Computer Vision tool for identifying material types in excavation images
Supports both image upload and live camera detection
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
import time
from pathlib import Path
from utils import (
    MaterialDetector, 
    visualize_detections, 
    draw_detection_summary,
    draw_fps_overlay,
    CameraStream,
    list_available_cameras,
    get_camera_info
)
import sys
import config



# Page configuration
st.set_page_config(
    page_title="Archaeological Material Detector",
    page_icon="🏺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #D4AF37;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .detection-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_detector():
    """Load and cache the YOLOv8 detector"""
    detector = MaterialDetector()
    with st.spinner("Loading detection model..."):
        if detector.load_model():
            return detector
        else:
            st.error("Failed to load model. Please check the model path.")
            return None


def process_image(image, detector):
    """
    Process image through detection pipeline
    
    Args:
        image: PIL Image or numpy array
        detector: MaterialDetector instance
        
    Returns:
        tuple: (annotated_image, detections, summary)
    """
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # Convert RGB to BGR for OpenCV
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image_np
    
    # Run detection
    detections = detector.detect(image_bgr)
    
    # Get summary
    summary = detector.get_detection_summary(detections)
    
    # Visualize detections
    annotated = visualize_detections(image_bgr, detections)
    
    # Add summary overlay
    annotated = draw_detection_summary(annotated, summary)
    
    # Convert back to RGB for display
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    return annotated_rgb, detections, summary


def process_camera_frame(frame, detector):
    """
    Process a single camera frame
    
    Args:
        frame: numpy array (BGR from camera)
        detector: MaterialDetector instance
        
    Returns:
        tuple: (annotated_frame, detections, summary)
    """
    # Run detection
    detections = detector.detect(frame)
    
    # Get summary
    summary = detector.get_detection_summary(detections)
    
    # Visualize detections
    annotated = visualize_detections(frame, detections)
    
    # Convert to RGB for Streamlit
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    return annotated_rgb, detections, summary


def main():
    """Main application function"""
    
    # Header
    st.markdown('<div class="main-header">🏺 Archaeological Material Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Identify Gold, Plastic, Stone, Soil, and Humans using AI</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model information
        st.subheader("Model Information")
        st.info("Using YOLOv8 for object detection")
        
        # Detection settings
        st.subheader("Detection Parameters")
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=config.CONFIDENCE_THRESHOLD,
            step=0.05,
            help="Minimum confidence score for detections"
        )
        
        # Update config
        config.CONFIDENCE_THRESHOLD = confidence
        
        # Model Selection
        st.subheader("🤖 Model Selection")
        
        # Find available models
        available_models = ["yolov8n.pt (Default - 80 classes)"]
        custom_model_path = Path("models/custom_model.pt")
        if custom_model_path.exists():
            available_models.append("custom_model.pt (Your Trained Model)")
        
        # Model selector
        selected_model_display = st.selectbox(
            "Choose Model",
            available_models,
            help="Switch between default and your trained model"
        )
        
        # Map selection to actual path
        if "custom_model.pt" in selected_model_display:
            selected_model = "models/custom_model.pt"
        else:
            selected_model = "yolov8n.pt"
        
        # Update config if changed
        if config.MODEL_PATH != selected_model:
            config.MODEL_PATH = selected_model
            st.cache_resource.clear()  # Clear cache to reload detector
            st.rerun()
        
        # Training Mode
        st.subheader("🏋️ Training Mode")
        st.info("Go to the **'🏋️ Train Model'** tab to create your own custom detector.")
        
        # About section
        st.subheader("About")
        st.markdown("""
        **YOLOv8 Object Detection App**
        - Real-time camera detection
        - Image analysis
        - Custom model support
        """)
    
    # Load detector
    detector = load_detector()
    
    if detector is None:
        st.error("⚠️ Failed to initialize the detector. Please check your setup.")
        return
        
    # Display current classes in sidebar
    with st.sidebar:
        st.subheader("Current Model Classes")
        class_list = detector.get_class_list()
        classes_str = ", ".join(list(class_list.values())[:10])
        if len(class_list) > 10:
            classes_str += f" +{len(class_list)-10} more"
        st.caption(f"Detecting {len(class_list)} classes: {classes_str}")
    
    # Load detector
    detector = load_detector()
    
    if detector is None:
        st.error("⚠️ Failed to initialize the detector. Please check your setup.")
        return
    
    # Create tabs for different modes
    tab1, tab2, tab3 = st.tabs(["📤 Image Upload", "📹 Live Camera", "🏋️ Train Model"])
    
    # ========== IMAGE UPLOAD TAB ==========
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📤 Upload Image")
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Choose an excavation image",
                type=config.SUPPORTED_FORMATS,
                help="Upload JPG, PNG, or BMP images"
            )
            
            # Sample images option
            st.markdown("---")
            st.caption("Or select a sample image")
            sample_dir = "sample_images"
            if os.path.exists(sample_dir):
                sample_files = [f for f in os.listdir(sample_dir) 
                              if f.split('.')[-1].lower() in config.SUPPORTED_FORMATS]
                if sample_files:
                    selected_sample = st.selectbox("Sample Images", ["None"] + sample_files)
                    if selected_sample != "None":
                        sample_path = os.path.join(sample_dir, selected_sample)
                        uploaded_file = sample_path
        
        with col2:
            st.subheader("🔍 Detection Results")
        
        # Process image if uploaded
        if uploaded_file is not None:
            try:
                # Load image
                if isinstance(uploaded_file, str):
                    # From sample directory
                    image = Image.open(uploaded_file)
                    image_path = uploaded_file
                else:
                    # From file uploader
                    image = Image.open(uploaded_file)
                    image_path = uploaded_file.name
                
                # Display original image
                with col1:
                    st.image(image, caption="Original Image", use_column_width=True)
                    
                    # Run detection button
                    run_detection = st.button("🚀 Run Detection", type="primary", use_container_width=True)
                
                # Run detection
                if run_detection or st.session_state.get('auto_detect', False):
                    with st.spinner("🔎 Analyzing image..."):
                        # Process image
                        annotated_image, detections, summary = process_image(image, detector)
                        
                        # Display results
                        with col2:
                            st.image(annotated_image, caption="Detected Materials", use_column_width=True)
                            
                            # Summary statistics
                            st.markdown('<div class="detection-box">', unsafe_allow_html=True)
                            st.metric("Total Detections", summary['total'])
                            
                            if summary['total'] > 0:
                                st.subheader("Materials Found:")
                                for class_name, count in summary['by_class'].items():
                                    col_a, col_b = st.columns([3, 1])
                                    with col_a:
                                        st.write(f"**{class_name}**")
                                    with col_b:
                                        st.write(f"{count}")
                            else:
                                st.info("No materials detected in this image.")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Download button
                            if summary['total'] > 0:
                                # Convert to PIL for download
                                result_pil = Image.fromarray(annotated_image)
                                buf = io.BytesIO()
                                result_pil.save(buf, format='PNG')
                                byte_im = buf.getvalue()
                                
                                st.download_button(
                                    label="📥 Download Annotated Image",
                                    data=byte_im,
                                    file_name=f"detected_{os.path.basename(image_path)}",
                                    mime="image/png",
                                    use_container_width=True
                                )
            
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
                st.exception(e)
        
        else:
            # Welcome message
            st.info("👆 Please upload an image or select a sample to begin detection.")
    
    # ========== LIVE CAMERA TAB ==========
    with tab2:
        st.subheader("📹 Live Camera Detection")
        
        # Camera controls
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.markdown("### Camera Controls")
            
            # Camera Source Selection
            camera_source = st.radio(
                "Camera Source", 
                ["Device Camera (Mobile)", "Server Camera (PC)"],
                help="Use 'Device Camera' for phones/tablets. Use 'Server Camera' for webcam attached to PC."
            )
            
            if camera_source == "Server Camera (PC)":
                # List available cameras
                available_cameras = list_available_cameras()
                
                if not available_cameras:
                    st.error("❌ No server cameras detected.")
                else:
                    camera_options = [f"Camera {i}" for i in available_cameras]
                    selected_camera_idx = st.selectbox(
                        "Select Server Camera",
                        range(len(available_cameras)),
                        format_func=lambda x: camera_options[x]
                    )
                    selected_camera = available_cameras[selected_camera_idx]
                    
                    # Start/Stop buttons
                    col_a, col_b = st.columns(2)
                    with col_a:
                        start_camera = st.button("▶️ Start Camera", type="primary", use_container_width=True)
                    with col_b:
                        stop_camera = st.button("⏹️ Stop Camera", type="secondary", use_container_width=True)
                    
                    if start_camera:
                        if st.session_state.camera_stream is not None:
                            st.session_state.camera_stream.stop()
                        
                        st.session_state.camera_stream = CameraStream(src=selected_camera)
                        if st.session_state.camera_stream.start():
                            st.session_state.camera_running = True
                            st.success("✅ Camera started!")
                        else:
                            st.error("❌ Failed to start camera.")
                    
                    if stop_camera:
                        if st.session_state.camera_stream is not None:
                            st.session_state.camera_stream.stop()
                            st.session_state.camera_running = False
                            st.info("⏹️ Camera stopped.")

            else:
                st.info("📱 On mobile? Click 'Take Photo' below to analyze a frame.")
                st.warning("⚠️ Live video on mobile uses data. Use Wi-Fi.")

        # Camera Display
        with col2:
            video_placeholder = st.empty()
            
            # --- SERVER SIDE CAMERA LOGIC ---
            if camera_source == "Server Camera (PC)":
                if st.session_state.camera_running and st.session_state.camera_stream is not None:
                    # Real-time detection loop
                    frame_count = 0
                    while st.session_state.camera_running:
                        frame = st.session_state.camera_stream.read()
                        if frame is not None:
                            frame_count += 1
                            if frame_count % config.DETECTION_SKIP_FRAMES == 0:
                                annotated_frame, detections, summary = process_camera_frame(frame, detector)
                                fps = st.session_state.camera_stream.get_fps()
                                annotated_frame = draw_fps_overlay(annotated_frame, fps, summary['total'])
                                video_placeholder.image(annotated_frame, channels="RGB", use_column_width=True)
                                fps_placeholder.metric("FPS", f"{fps:.1f}")
                                detections_placeholder.metric("Live Detections", summary['total'])
                            time.sleep(0.01)
                        if stop_camera or not st.session_state.camera_stream.is_running():
                            break
                else:
                    video_placeholder.info("📹 Click 'Start Camera' to begin live detection")

            # --- CLIENT SIDE CAMERA LOGIC (MOBILE) ---
            else:
                # Use Streamlit's built-in camera input for mobile
                img_file_buffer = st.camera_input("Take a picture")
                
                if img_file_buffer is not None:
                    # Convert to CV2 image
                    bytes_data = img_file_buffer.getvalue()
                    cv2_img = cvpre_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    
                    # Run detection
                    annotated_frame, detections, summary = process_camera_frame(cv2_img, detector)
                    
                    # Show result
                    st.image(annotated_frame, channels="RGB", caption="Processed Frame")
                    
                    # Show metrics
                    detections_placeholder.metric("Detections", summary['total'])
                    st.success(f"Found {summary['total']} objects!")




    # ========== TRAINING TAB ==========
    with tab3:
        st.header("🏋️ Train Custom Model")
        st.markdown("Create a model that recognizes **YOUR** objects.")

        # Initialize dataset paths
        DATASET_DIR = Path("dataset")
        for split in ['train', 'val', 'test']:
            (DATASET_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
            (DATASET_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        
        # Initialize session state with persistence
        if 'class_names' not in st.session_state:
            # Try to load from existing data.yaml
            if Path("data.yaml").exists():
                try:
                    import yaml
                    with open("data.yaml", 'r') as f:
                        data_config = yaml.safe_load(f)
                        if 'names' in data_config:
                            st.session_state.class_names = list(data_config['names'].values())
                        else:
                            st.session_state.class_names = []
                except:
                    st.session_state.class_names = []
            else:
                st.session_state.class_names = []

        col_t1, col_t2 = st.columns([1, 1])

        with col_t1:
            st.subheader("1. Create Label & Upload")
            
            # 1. Add Label
            new_label = st.text_input("Object Name (e.g. 'Gold Coin')")
            if st.button("➕ Add Class") and new_label:
                if new_label not in st.session_state.class_names:
                    st.session_state.class_names.append(new_label)
                    st.success(f"Added: {new_label}")
            
            # Show current classes
            if st.session_state.class_names:
                st.caption(f"Classes: {', '.join(st.session_state.class_names)}")
                
                # 2. Upload & Label
                selected_class = st.selectbox("Select Class to Upload For:", st.session_state.class_names)
                target_id = st.session_state.class_names.index(selected_class)
                
                uploaded_train_files = st.file_uploader(f"Upload Images for '{selected_class}'", 
                                                      type=['jpg','png','jpeg'], accept_multiple_files=True)
                
                if uploaded_train_files:
                    if st.button(f"💾 Save & Auto-Label {len(uploaded_train_files)} Images"):
                        count = 0
                        for file in uploaded_train_files:
                            # Save Image
                            img_path = DATASET_DIR / 'images' / 'train' / file.name
                            if img_path.exists():
                                import uuid
                                img_path = DATASET_DIR / 'images' / 'train' / f"{file.name.split('.')[0]}_{uuid.uuid4().hex[:4]}.jpg"
                            
                            with open(img_path, "wb") as f:
                                f.write(file.getbuffer())
                            
                            # Create Label (Auto - Full Box)
                            label_path = DATASET_DIR / 'labels' / 'train' / f"{img_path.stem}.txt"
                            with open(label_path, "w") as f:
                                f.write(f"{target_id} 0.5 0.5 0.95 0.95")
                            count += 1
                        st.success(f"✅ Saved & Labeled {count} images!")

        with col_t2:
            st.subheader("2. Start Training")
            
            if st.session_state.class_names:
                # Count total images
                train_img_dir = DATASET_DIR / 'images' / 'train'
                total_images = len(list(train_img_dir.glob('*.jpg'))) + len(list(train_img_dir.glob('*.png'))) + len(list(train_img_dir.glob('*.jpeg')))
                
                st.info(f"Ready to train on {len(st.session_state.class_names)} classes with {total_images} images.")
                
                # Validate minimum images
                if total_images < 10:
                    st.warning(f"⚠️ You need at least 10 images to train. You currently have {total_images}.")
                    st.info("💡 Upload more images above to reach the minimum.")
                
                if st.button("🚀 Start Training Now", type="primary", disabled=(total_images < 10)):
                    # Generate Config
                    yaml = f"path: {DATASET_DIR.absolute()}\ntrain: images/train\nval: images/train\nnc: {len(st.session_state.class_names)}\nnames:\n"
                    for i, name in enumerate(st.session_state.class_names):
                        yaml += f"  {i}: {name}\n"
                    with open("data.yaml", "w") as f:
                        f.write(yaml)
                    
                    st.toast("Config created. Starting training...")
                    
                    # Run Training
                    cmd = [sys.executable, "train_model.py"]
                    
                    try:
                        import subprocess
                        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
                        
                        log_area = st.empty()
                        logs = ""
                        
                        st.subheader("Training Logs:")
                        for line in process.stdout:
                            logs += line
                            # Keep only last 20 lines to prevent UI lag
                            display_logs = "\n".join(logs.split("\n")[-20:])
                            log_area.code(display_logs)
                        
                        # Wait for process to complete and check return code
                        return_code = process.wait()
                        
                        if return_code == 0:
                            st.success("✅ Training Complete! Check `runs/detect/custom_model/weights/best.pt`")
                            st.balloons()
                        else:
                            st.error(f"❌ Training failed with exit code {return_code}. Check logs above.")
                        
                    except Exception as e:
                        st.error(f"❌ Training failed: {e}")
            else:
                st.warning("👈 Add at least one class first.")
        
        # Model Loading Section
        st.divider()
        st.subheader("3. Use Your Trained Model")
        
        trained_model_path = Path("runs/detect/custom_model/weights/best.pt")
        if trained_model_path.exists():
            st.success(f"✅ Found trained model: `{trained_model_path}`")
            
            if st.button("🔄 Load My Trained Model", type="secondary"):
                try:
                    import shutil
                    
                    # Copy model to models folder
                    target_path = Path("models/custom_model.pt")
                    shutil.copy(trained_model_path, target_path)
                    
                    # Update config.py
                    config_path = Path("config.py")
                    with open(config_path, 'r') as f:
                        config_content = f.read()
                    
                    # Replace MODEL_PATH line
                    config_content = config_content.replace(
                        'MODEL_PATH = "yolov8n.pt"',
                        'MODEL_PATH = "models/custom_model.pt"'
                    )
                    
                    with open(config_path, 'w') as f:
                        f.write(config_content)
                    
                    st.success("✅ Model loaded! Restarting app...")
                    st.info("🔄 The page will reload automatically.")
                    time.sleep(2)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Failed to load model: {e}")
        else:
            st.info("Train a model first to see it here.")

if __name__ == "__main__":
    main()
