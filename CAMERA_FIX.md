# Camera Permission Fix Guide

## Problem: "Failed to start camera. Check permissions."

**Error Code**: `-1072875772` (MSMF error - Media Foundation permission denied)

This means Windows is **blocking camera access** for Python/desktop applications.

---

## ✅ SOLUTION (Step-by-Step)

### Step 1: Enable Camera Permissions in Windows

1. **Open Windows Settings**
   - Press `Windows + I`
   - Or search for "Settings" in Start menu

2. **Navigate to Camera Privacy**
   - Go to: **Privacy & Security** → **Camera**
   
3. **Enable Camera Access**
   - Turn ON: **"Camera access"** (top toggle)
   - Turn ON: **"Let apps access your camera"**
   - Turn ON: **"Let desktop apps access your camera"** ⚠️ CRITICAL

4. **Verify Python Access**
   - Scroll down to see list of apps
   - Make sure Python is allowed

### Step 2: Close Other Camera Apps

**Close these if running:**
- Zoom
- Microsoft Teams
- Skype
- OBS Studio
- Any video recording software
- Windows Camera app

### Step 3: Restart Terminal

1. Close PowerShell/Terminal completely
2. Open new PowerShell/Terminal
3. Try running the app again

### Step 4: Test Camera

```bash
# Run the diagnostic tool
py -3.11 test_camera.py
```

Expected output if fixed:
```
[OK] Camera 0 is working!
Resolution: 640x480
FPS: 30
```

### Step 5: Run Application

```bash
py -3.11 -m streamlit run app.py
```

---

## Alternative Solution: Try DirectShow Backend

If Media Foundation (MSMF) doesn't work, try DirectShow backend:

**Edit `utils/camera.py` line 29:**

```python
# OLD:
self.stream = cv2.VideoCapture(camera_index)

# NEW (DirectShow):
self.stream = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
```

This forces OpenCV to use DirectShow instead of Media Foundation.

---

## Quick Verification Commands

**Test 1: Check if camera device exists**
```powershell
Get-PnpDevice -FriendlyName *camera* | Select-Object Status, FriendlyName
```

**Test 2: Quick Python camera test**
```bash
py -3.11 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.read()[0] else 'Camera FAIL'); cap.release()"
```

**Test 3: Windows Camera app**
- Open "Camera" app from Start menu
- If it works → Permission issue with Python
- If it doesn't work → Hardware/driver issue

---

## Common Issues

### Issue 1: "Let desktop apps access your camera" is GRAYED OUT

**Solution:**
1. Press `Windows + R`
2. Type: `gpedit.msc` (Group Policy Editor)
3. Navigate to: **Computer Configuration** → **Administrative Templates** → **Windows Components** → **Camera**
4. Double-click "Allow Use of Camera"
5. Set to **"Not Configured"** or **"Enabled"**
6. Click OK, restart computer

### Issue 2: Camera works in Windows Camera app but not in Python

**Solution:**
- This is definitely a permissions issue
- Make sure "Let **desktop apps** access your camera" is ON
- Restart your Terminal completely
- Try running as Administrator

### Issue 3: USB camera not detected

**Solutions:**
- Unplug and replug the USB camera
- Try different USB port
- Check Device Manager for driver issues
- Update camera drivers

---

## If Nothing Works: Use Image Mode

The application still works perfectly in **Image Upload mode**!

1. Open app: `py -3.11 -m streamlit run app.py`
2. Click "Image Upload" tab
3. Upload photos instead of using live camera
4. Get the same detection results

---

## Need More Help?

Run the diagnostic tool:
```bash
py -3.11 test_camera.py
```

This will tell you exactly what's wrong with your camera setup.

---

**Most Common Fix:**
Windows Settings → Privacy & Security → Camera → Turn ON "Let desktop apps access your camera"
