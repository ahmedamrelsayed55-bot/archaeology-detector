"""
Quick installation script to set up the environment
"""
import subprocess
import sys

def main():
    print("=" * 60)
    print("Archaeological Material Detection Application")
    print("Installation Script")
    print("=" * 60)
    print()
    
    # Install dependencies
    print("Installing dependencies...")
    print("This may take a few minutes...")
    print()
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"
        ])
        print("✓ Dependencies installed successfully!")
        print()
        print("=" * 60)
        print("Installation complete!")
        print()
        print("To run the application:")
        print("  streamlit run app.py")
        print()
        print("Or with specific Python version:")
        print("  py -3.11 -m streamlit run app.py")
        print("=" * 60)
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing dependencies: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
