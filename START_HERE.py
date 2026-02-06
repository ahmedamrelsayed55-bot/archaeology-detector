"""
Quick start guide for the Archaeological Material Detection Application
"""

print("""
========================================================================
    Archaeological Material Detection Application
========================================================================

QUICK START GUIDE
========================================================================

Step 1: Install Dependencies
========================================================================
Run one of the following commands:

    Option A - Using install script:
        py -3.11 install.py
    
    Option B - Direct installation:
        py -3.11 -m pip install -r requirements.txt

Step 2: Run the Application
========================================================================
Once dependencies are installed:

    py -3.11 -m streamlit run app.py

The application will open in your default browser at:
    http://localhost:8501

Step 3: Use the Application
========================================================================
1. Upload an image (JPG, PNG, BMP)
2. Click "Run Detection"
3. View annotated results
4. Download processed images

========================================================================

KEY FEATURES:
+ YOLOv8-powered object detection
+ OpenCV image processing
+ Real-time material classification (Metal/Plastic)
+ Professional bounding box visualization
+ Confidence scores display
+ Downloadable results

IMPORTANT NOTES:
* First run will download YOLOv8n model (~6MB)
* For production use, train a custom model with your data
* Adjust confidence threshold in sidebar for sensitivity

TROUBLESHOOTING:
========================================================================

Issue: Module not found errors
Solution: Reinstall dependencies
    py -3.11 -m pip install -r requirements.txt --upgrade

Issue: Port already in use
Solution: Use different port
    py -3.11 -m streamlit run app.py --server.port 8502

Issue: GPU not detected (optional)
Solution: Install PyTorch with CUDA if you have NVIDIA GPU
    py -3.11 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

Need help? Check README.md for detailed documentation.

========================================================================
""")
