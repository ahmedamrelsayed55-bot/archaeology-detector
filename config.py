"""
Configuration settings for YOLOv8 Object Detection Application
Generic configuration supporting both pretrained and custom models
"""
import os

# Model Configuration
MODEL_PATH = "yolov8n.pt"  # Default to pretrained YOLOv8n (80 COCO classes)
# To use custom trained model, change to: "models/custom_model.pt"

CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence for detections
IOU_THRESHOLD = 0.45  # IoU threshold for NMS

# Application Settings
MAX_IMAGE_SIZE = 1280  # Max dimension for processing
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "bmp"]

# Camera Settings - Flexible (auto-detect from camera)
CAMERA_DEFAULT_INDEX = 0  # Default camera device
# Camera resolution and FPS will auto-detect from available camera
# Override these only if needed:
# CAMERA_WIDTH = 640
# CAMERA_HEIGHT = 480
# CAMERA_FPS_TARGET = 30

DETECTION_SKIP_FRAMES = 1  # Process every N frames (1 = process all frames)

# Performance Settings
ENABLE_GPU = True  # Use GPU if available
MAX_DETECTIONS = 100  # Maximum detections per frame

# Create necessary directories
os.makedirs("models", exist_ok=True)
os.makedirs("sample_images", exist_ok=True)
