#!/usr/bin/env python3
"""
Setup script for Multi-Camera Tracking System

This script helps users set up the tracking system by:
- Installing dependencies
- Creating necessary directories
- Downloading YOLO models
- Validating the installation

Usage: python setup.py [--dev]
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path
import argparse


def run_command(command, description=""):
    """Run a shell command and handle errors"""
    print(f"Running: {description or command}")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False


def create_directories():
    """Create necessary directories"""
    directories = [
        "data",
        "logs", 
        "config",
        "models",
        "src/detectors",
        "src/trackers", 
        "src/utils",
        "src/calibration"
    ]
    
    print("Creating directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")


def install_dependencies(dev=False):
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    
    # Install main requirements
    if not run_command("pip install -r requirements.txt", 
                      "Installing main dependencies"):
        return False
    
    # Install development dependencies if requested
    if dev:
        dev_packages = [
            "pytest>=6.0.0",
            "black>=21.0.0", 
            "flake8>=3.9.0",
            "mypy>=0.910"
        ]
        
        for package in dev_packages:
            if not run_command(f"pip install {package}", 
                              f"Installing {package}"):
                print(f"Warning: Failed to install {package}")
    
    return True


def download_yolo_models():
    """Download YOLO model files"""
    models = {
        "yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt",
        "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    }
    
    print("Downloading YOLO models...")
    
    for model_name, url in models.items():
        model_path = Path(model_name)
        
        if model_path.exists():
            print(f"  ✓ {model_name} already exists")
            continue
        
        try:
            print(f"  Downloading {model_name}...")
            urllib.request.urlretrieve(url, model_name)
            print(f"  ✓ Downloaded {model_name}")
        except Exception as e:
            print(f"  ✗ Failed to download {model_name}: {e}")
            print(f"    You can download manually from: {url}")


def validate_installation():
    """Validate the installation"""
    print("Validating installation...")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("  ✗ Python 3.7+ required")
        return False
    else:
        print(f"  ✓ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check key dependencies
    required_packages = [
        "numpy",
        "opencv-python", 
        "torch",
        "ultralytics",
        "scipy"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} not found")
            missing_packages.append(package)
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available (GPU: {torch.cuda.get_device_name(0)})")
        else:
            print("  ⚠ CUDA not available (will use CPU)")
    except ImportError:
        pass
    
    # Check YOLO models
    yolo_models = ["yolo11n.pt", "yolov8n.pt"]
    model_found = False
    for model in yolo_models:
        if Path(model).exists():
            print(f"  ✓ {model}")
            model_found = True
            break
    
    if not model_found:
        print("  ✗ No YOLO models found")
        missing_packages.append("YOLO models")
    
    # Check directories
    required_dirs = ["data", "logs", "config", "src"]
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"  ✓ {directory}/ directory")
        else:
            print(f"  ✗ {directory}/ directory missing")
    
    if missing_packages:
        print(f"\nMissing components: {', '.join(missing_packages)}")
        return False
    
    print("\n✓ Installation validation successful!")
    return True


def show_usage_examples():
    """Show usage examples"""
    print("\n" + "="*60)
    print("SETUP COMPLETE - Usage Examples")
    print("="*60)
    
    examples = [
        ("Basic tracking", "python main.py track video1.mp4 video2.mp4"),
        ("Batch processing", "python main.py track video1.mp4 video2.mp4 --mode batch"),
        ("Calculate homography", "python main.py calibrate homography video1.mp4 video2.mp4"),
        ("Create view union", "python main.py union video1.mp4 video2.mp4"),
        ("System info", "python main.py info --check-deps"),
        ("Help", "python main.py --help")
    ]
    
    for description, command in examples:
        print(f"\n{description}:")
        print(f"  {command}")
    
    print(f"\nFor more information, see README.md")
    print("="*60)


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup Multi-Camera Tracking System")
    parser.add_argument("--dev", action="store_true", 
                       help="Install development dependencies")
    parser.add_argument("--skip-models", action="store_true",
                       help="Skip downloading YOLO models")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate existing installation")
    
    args = parser.parse_args()
    
    print("Multi-Camera Tracking System Setup")
    print("="*40)
    
    if args.validate_only:
        success = validate_installation()
        sys.exit(0 if success else 1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies(args.dev):
        print("Failed to install dependencies")
        sys.exit(1)
    
    # Download YOLO models
    if not args.skip_models:
        download_yolo_models()
    
    # Validate installation
    if validate_installation():
        show_usage_examples()
        print("\n🎉 Setup completed successfully!")
    else:
        print("\n❌ Setup completed with issues. Please check the validation results above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
