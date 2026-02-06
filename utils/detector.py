"""
YOLOv8 Object Detector
Handles model loading, inference, and result processing
Supports both pretrained COCO models and custom trained models
"""
import cv2
import numpy as np
from ultralytics import YOLO
import config


class MaterialDetector:
    """Generic YOLOv8 object detector"""
    
    def __init__(self, model_path=None):
        """
        Initialize the detector
        
        Args:
            model_path (str): Path to YOLOv8 model weights
        """
        self.model_path = model_path or config.MODEL_PATH
        self.model = None
        self.confidence_threshold = config.CONFIDENCE_THRESHOLD
        self.iou_threshold = config.IOU_THRESHOLD
        
    def load_model(self):
        """
        Load YOLOv8 model
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.model = YOLO(self.model_path)
            print(f"Model loaded successfully: {self.model_path}")
            
            # Print model info
            if hasattr(self.model, 'names'):
                num_classes = len(self.model.names)
                print(f"Model has {num_classes} classes")
                
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def detect(self, image):
        """
        Run detection on image
        
        Args:
            image: Input image (numpy array, BGR format)
            
        Returns:
            list: List of detections, each containing:
                  {bbox, confidence, class_id, class_name}
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Run inference
        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )[0]
        
        # Process detections
        detections = []
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()  # Bounding boxes
            confidences = results.boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = results.boxes.cls.cpu().numpy().astype(int)  # Class IDs
            
            for bbox, conf, class_id in zip(boxes, confidences, class_ids):
                detection = {
                    'bbox': bbox.tolist(),  # [x1, y1, x2, y2]
                    'confidence': float(conf),
                    'class_id': int(class_id),
                    'class_name': self._get_class_name(class_id)
                }
                detections.append(detection)
        
        return detections
    
    def _get_class_name(self, class_id):
        """
        Get class name from class ID
        Uses the model's native class names
        
        Args:
            class_id (int): Class ID from model
            
        Returns:
            str: Class name
        """
        if hasattr(self.model, 'names'):
            return self.model.names.get(class_id, f"Class_{class_id}")
        return f"Class_{class_id}"
    
    def get_detection_summary(self, detections):
        """
        Generate summary statistics from detections
        
        Args:
            detections (list): List of detection dictionaries
            
        Returns:
            dict: Summary with total count and count by class
        """
        summary = {
            'total': len(detections),
            'by_class': {}
        }
        
        for det in detections:
            class_name = det['class_name']
            summary['by_class'][class_name] = summary['by_class'].get(class_name, 0) + 1
        
        return summary
    
    def get_class_list(self):
        """
        Get list of all classes the model can detect
        
        Returns:
            dict: Dictionary of class_id: class_name
        """
        if hasattr(self.model, 'names'):
            return self.model.names
        return {}
