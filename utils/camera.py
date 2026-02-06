"""
Camera Stream Module
Handles video capture from camera with auto-detection of capabilities
"""
import cv2
import threading
import time


def list_available_cameras(max_test=5):
    """
    List all available camera indices
    
    Args:
        max_test (int): Maximum number of indices to test
        
    Returns:
        list: List of available camera indices
    """
    available = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def get_camera_info(camera_index):
    """
    Get camera information (resolution, FPS capabilities)
    
    Args:
        camera_index (int): Camera index
        
    Returns:
        dict: Camera information or None if camera not available
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None
    
    info = {
        'index': camera_index,
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': int(cap.get(cv2.CAP_PROP_FPS)),
        'backend': cap.getBackendName()
    }
    
    cap.release()
    return info


class CameraStream:
    """Threaded camera stream for efficient real-time capture"""
    
    def __init__(self, camera_index=0, width=None, height=None):
        """
        Initialize camera stream with auto-detection
        
        Args:
            camera_index (int): Camera device index
            width (int): Optional target width (None = use camera default)
            height (int): Optional target height (None = use camera default)
        """
        self.camera_index = camera_index
        self.stream = None
        self.frame = None
        self.running = False
        self.thread = None
        
        # FPS tracking
        self.frame_count = 0
        self.start_time = None
        self.fps = 0
        
        # Optional resolution override
        self.target_width = width
        self.target_height = height
        
    def start(self):
        """
        Start the camera stream
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Open camera
            self.stream = cv2.VideoCapture(self.camera_index)
            
            if not self.stream.isOpened():
                print(f"Failed to open camera {self.camera_index}")
                return False
            
            # Set resolution if specified
            if self.target_width and self.target_height:
                self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            
            # Get actual resolution
            actual_width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Camera resolution: {actual_width}x{actual_height}")
            
            # Start capture thread
            self.running = True
            self.start_time = time.time()
            self.thread = threading.Thread(target=self._update_frame, daemon=True)
            self.thread.start()
            
            return True
            
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False
    
    def _update_frame(self):
        """Background thread to continuously read frames"""
        while self.running:
            if self.stream and self.stream.isOpened():
                ret, frame = self.stream.read()
                if ret:
                    self.frame = frame
                    self.frame_count += 1
                    
                    # Calculate FPS every second
                    elapsed = time.time() - self.start_time
                    if elapsed > 0:
                        self.fps = self.frame_count / elapsed
            else:
                break
    
    def read(self):
        """
        Read the latest frame
        
        Returns:
            numpy array: Latest frame or None if not available
        """
        return self.frame
    
    def get_fps(self):
        """
        Get current FPS
        
        Returns:
            float: Current FPS
        """
        return self.fps
    
    def is_running(self):
        """Check if stream is running"""
        return self.running
    
    def stop(self):
        """Stop the camera stream"""
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=1.0)
        
        if self.stream:
            self.stream.release()
            self.stream = None
        
        self.frame = None
        print("Camera stopped")
