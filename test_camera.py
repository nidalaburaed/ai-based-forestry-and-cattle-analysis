"""
Camera diagnostic tool to test webcam access with different OpenCV backends.
Run this to diagnose camera access issues.
"""
import cv2
import sys

def test_camera_backend(camera_index, backend_name, backend_flag=None):
    """Test opening camera with a specific backend."""
    print(f"\nTesting {backend_name}...")
    try:
        if backend_flag is not None:
            cap = cv2.VideoCapture(camera_index, backend_flag)
        else:
            cap = cv2.VideoCapture(camera_index)
        
        if cap.isOpened():
            # Try to read a frame
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"  ✓ SUCCESS: Camera opened with {backend_name}")
                print(f"    Resolution: {w}x{h}")
                
                # Get camera properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"    FPS: {fps}")
                
                cap.release()
                return True
            else:
                print(f"  ✗ FAILED: Camera opened but couldn't read frame with {backend_name}")
                cap.release()
                return False
        else:
            print(f"  ✗ FAILED: Cannot open camera with {backend_name}")
            cap.release()
            return False
    except Exception as e:
        print(f"  ✗ ERROR with {backend_name}: {e}")
        return False

def main():
    print("=" * 60)
    print("OpenCV Camera Diagnostic Tool")
    print("=" * 60)
    
    # Get camera index from command line or use default
    camera_index = 0
    if len(sys.argv) > 1:
        try:
            camera_index = int(sys.argv[1])
        except ValueError:
            print(f"Invalid camera index: {sys.argv[1]}, using default: 0")
    
    print(f"\nTesting camera index: {camera_index}")
    print(f"OpenCV version: {cv2.__version__}")
    
    # Test different backends
    backends = [
        ("DirectShow (CAP_DSHOW)", cv2.CAP_DSHOW),
        ("Media Foundation (CAP_MSMF)", cv2.CAP_MSMF),
        ("Default Backend", None),
    ]
    
    success_count = 0
    for backend_name, backend_flag in backends:
        if test_camera_backend(camera_index, backend_name, backend_flag):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {success_count}/{len(backends)} backends successful")
    print("=" * 60)
    
    if success_count == 0:
        print("\n⚠ TROUBLESHOOTING TIPS:")
        print("1. Check if another application is using the camera")
        print("2. Check Windows camera privacy settings:")
        print("   Settings > Privacy > Camera > Allow apps to access camera")
        print("3. Try a different camera index (e.g., python test_camera.py 1)")
        print("4. Restart your computer")
        print("5. Update your webcam drivers")
        print("6. Try running as administrator")
    else:
        print("\n✓ Camera access is working!")
        print("  The application should now be able to access your camera.")

if __name__ == "__main__":
    main()

