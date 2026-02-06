"""
OpenCV Visualization Module
Functions for drawing bounding boxes, labels, and detection summaries
"""
import cv2
import numpy as np


# Default color palette for different classes
DEFAULT_COLORS = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 128),  # Purple
    (255, 165, 0),  # Orange
    (0, 128, 128),  # Teal
    (128, 128, 0),  # Olive
]


def get_color_for_class(class_id):
    """Get a consistent color for a class ID"""
    return DEFAULT_COLORS[class_id % len(DEFAULT_COLORS)]


def visualize_detections(image, detections):
    """
    Draw bounding boxes and labels on the image
    
    Args:
        image: Input image (numpy array)
        detections: List of detection dictionaries with 'bbox', 'confidence', 'class_name', 'class_id'
        
    Returns:
        numpy array: Annotated image
    """
    result_image = image.copy()
    
    for detection in detections:
        bbox = detection['bbox']
        confidence = detection['confidence']
        class_name = detection['class_name']
        class_id = detection.get('class_id', 0)
        
        # Extract coordinates
        x1, y1, x2, y2 = map(int, bbox)
        
        # Get color for this class
        color = get_color_for_class(class_id)
        
        # Draw bounding box
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label text
        label = f"{class_name}: {confidence:.2f}"
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        
        # Draw label background rectangle
        cv2.rectangle(
            result_image,
            (x1, y1 - text_height - baseline - 10),
            (x1 + text_width + 10, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            result_image,
            label,
            (x1 + 5, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),  # White text
            2,
            cv2.LINE_AA
        )
    
    return result_image


def draw_detection_summary(image, summary):
    """
    Draw detection summary overlay on image
    
    Args:
        image: Input image (numpy array)
        summary: Summary dictionary with 'total' and 'by_class'
        
    Returns:
        numpy array: Image with summary overlay
    """
    result_image = image.copy()
    h, w = result_image.shape[:2]
    
    # Create semi-transparent overlay
    overlay = result_image.copy()
    
    # Summary box dimensions (bottom-left corner)
    box_width = 250
    box_height = 60 + (len(summary['by_class']) * 25)
    box_x = 10
    box_y = h - box_height - 10
    
    # Draw background
    cv2.rectangle(
        overlay,
        (box_x, box_y),
        (box_x + box_width, box_y + box_height),
        (40, 40, 40),
        -1
    )
    
    # Blend with original image
    cv2.addWeighted(overlay, 0.7, result_image, 0.3, 0, result_image)
    
    # Draw border
    cv2.rectangle(
        result_image,
        (box_x, box_y),
        (box_x + box_width, box_y + box_height),
        (255, 255, 255),
        2
    )
    
    # Draw title
    cv2.putText(
        result_image,
        "Detection Summary",
        (box_x + 10, box_y + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    
    # Draw total count
    cv2.putText(
        result_image,
        f"Total: {summary['total']}",
        (box_x + 10, box_y + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )
    
    # Draw class counts
    y_offset = 75
    for idx, (class_name, count) in enumerate(summary['by_class'].items()):
        color = get_color_for_class(idx)
        
        # Draw colored indicator
        cv2.circle(result_image, (box_x + 15, box_y + y_offset - 5), 5, color, -1)
        
        # Draw class name and count
        text = f"{class_name}: {count}"
        cv2.putText(
            result_image,
            text,
            (box_x + 30, box_y + y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )
        
        y_offset += 25
    
    return result_image


def draw_fps_overlay(image, fps, detections_count=0):
    """
    Draw FPS and performance metrics overlay on image
    
    Args:
        image: Input image (numpy array)
        fps (float): Current FPS
        detections_count (int): Number of detections in current frame
        
    Returns:
        numpy array: Image with FPS overlay
    """
    result_image = image.copy()
    
    # Create semi-transparent background for metrics
    overlay = result_image.copy()
    
    # Metrics box dimensions (top-left corner)
    box_width = 200
    box_height = 70
    box_x = 10
    box_y = 10
    
    # Draw background
    cv2.rectangle(
        overlay,
        (box_x, box_y),
        (box_x + box_width, box_y + box_height),
        (40, 40, 40),
        -1
    )
    
    # Blend with original
    cv2.addWeighted(overlay, 0.6, result_image, 0.4, 0, result_image)
    
    # Draw border
    cv2.rectangle(
        result_image,
        (box_x, box_y),
        (box_x + box_width, box_y + box_height),
        (0, 255, 0),
        2
    )
    
    # Draw FPS
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(
        result_image,
        fps_text,
        (box_x + 10, box_y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )
    
    # Draw detections count
    det_text = f"Detections: {detections_count}"
    cv2.putText(
        result_image,
        det_text,
        (box_x + 10, box_y + 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )
    
    return result_image
