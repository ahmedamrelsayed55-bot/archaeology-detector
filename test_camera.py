"""
Camera Diagnostic Tool
Tests camera access and provides troubleshooting information
"""
import cv2
import sys

def test_camera(index=0):
    """Test camera access at given index"""
    print(f"\n{'='*70}")
    print(f"Testing Camera {index}")
    print(f"{'='*70}")
    
    try:
        # Try to open camera
        cap = cv2.VideoCapture(index)
        
        if not cap.isOpened():
            print(f"[X] Failed to open camera {index}")
            print("\nPossible reasons:")
            print("  1. Camera is being used by another application")
            print("  2. Camera permissions not granted")
            print("  3. Camera driver issue")
            print("  4. No camera at this index")
            return False
        
        # Try to read a frame
        ret, frame = cap.read()
        
        if not ret:
            print(f"[X] Camera {index} opened but cannot read frames")
            print("\nPossible reasons:")
            print("  1. Camera hardware issue")
            print("  2. Driver problem")
            print("  3. PERMISSION DENIED - Check Windows camera settings")
            cap.release()
            return False
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        backend = cap.getBackendName()
        
        print(f"[OK] Camera {index} is working!")
        print(f"\nCamera Properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Backend: {backend}")
        print(f"  Frame shape: {frame.shape}")
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"[X] Error testing camera {index}: {e}")
        return False


def diagnose_cameras():
    """Diagnose all available cameras"""
    print("\n" + "="*70)
    print("CAMERA DIAGNOSTIC TOOL")
    print("="*70)
    print("\nOpenCV Version:", cv2.__version__)
    print("Python Version:", sys.version)
    
    # Test first 5 camera indices
    working_cameras = []
    
    for i in range(5):
        if test_camera(i):
            working_cameras.append(i)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if working_cameras:
        print(f"\n[OK] Found {len(working_cameras)} working camera(s): {working_cameras}")
        print("\nYou can use these cameras in the application!")
    else:
        print("\n[X] No working cameras found")
        print("\nTroubleshooting Steps:")
        print("\n1. CHECK CAMERA PERMISSIONS (Windows):")
        print("   - Open Settings")
        print("   - Go to Privacy & Security > Camera")
        print("   - Enable 'Let apps access your camera'")
        print("   - Enable 'Let desktop apps access your camera'")
        print("   - Scroll down and enable for Python")
        
        print("\n2. CLOSE OTHER APPLICATIONS:")
        print("   - Close Zoom, Teams, Skype, etc.")
        print("   - Close any other apps using the camera")
        
        print("\n3. CHECK CAMERA CONNECTION:")
        print("   - Make sure camera is plugged in (for external cameras)")
        print("   - Try unplugging and replugging USB camera")
        
        print("\n4. UPDATE CAMERA DRIVERS:")
        print("   - Open Device Manager")
        print("   - Find your camera under 'Cameras' or 'Imaging devices'")
        print("   - Right-click > Update driver")
        
        print("\n5. TEST WITH CAMERA APP:")
        print("   - Open Windows Camera app")
        print("   - If it works there, issue is with Python/OpenCV permissions")
        
        print("\n6. RUN AS ADMINISTRATOR:")
        print("   - Right-click PowerShell/Terminal")
        print("   - Select 'Run as administrator'")
        print("   - Try running the app again")
        
        print("\n7. SEE CAMERA_FIX.md for detailed solutions")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    diagnose_cameras()
