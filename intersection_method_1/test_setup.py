#!/usr/bin/env python3
"""
Test script to verify the refactored codebase works correctly

This script performs basic import tests and functionality checks
to ensure the refactoring was successful.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_imports():
    """Test that all modules can be imported correctly"""
    print("Testing imports...")
    
    try:
        # Test detector imports
        from src.detectors.yolo_detector import YOLOv8Detector
        print("  ✓ YOLOv8Detector imported")
        
        # Test tracker imports  
        from src.trackers.multi_camera_tracker import GlobalIDManager
        from src.trackers.yolo_multicam_tracker import YoloMultiCameraTracker
        print("  ✓ Tracker modules imported")
        
        # Test utility imports
        from src.utils.config import ConfigManager, get_config
        from src.utils.database import TrackingDatabase
        print("  ✓ Utility modules imported")
        
        return True
        
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False


def test_config_system():
    """Test configuration system"""
    print("Testing configuration system...")
    
    try:
        from src.utils.config import ConfigManager
        
        # Create config manager
        config_manager = ConfigManager()
        
        # Test default config
        config = config_manager.config
        assert config.detection.confidence_threshold == 0.6
        assert config.tracking.distance_threshold == 50.0
        
        # Test validation
        is_valid = config_manager.validate_config()
        print(f"  ✓ Configuration validation: {is_valid}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Configuration test error: {e}")
        return False


def test_database():
    """Test database functionality"""
    print("Testing database system...")
    
    try:
        from src.utils.database import TrackingDatabase, SessionRecord
        
        # Create test database
        db = TrackingDatabase("test_db.db")
        
        # Create test session
        session = SessionRecord(
            session_id="test_001",
            video1_path="test1.mp4", 
            video2_path="test2.mp4"
        )
        
        # Test session creation
        session_id = db.create_session(session)
        print(f"  ✓ Created test session: {session_id}")
        
        # Clean up
        os.remove("test_db.db")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Database test error: {e}")
        return False


def test_directory_structure():
    """Test that directory structure is correct"""
    print("Testing directory structure...")
    
    required_dirs = [
        "src",
        "src/detectors", 
        "src/trackers",
        "src/utils",
        "src/calibration",
        "config",
        "data",
        "logs"
    ]
    
    missing_dirs = []
    for directory in required_dirs:
        if not Path(directory).exists():
            missing_dirs.append(directory)
        else:
            print(f"  ✓ {directory}/")
    
    if missing_dirs:
        print(f"  ✗ Missing directories: {missing_dirs}")
        return False
    
    return True


def test_main_script():
    """Test main script functionality"""
    print("Testing main script...")
    
    try:
        # Test that main.py can be imported
        import main
        print("  ✓ main.py imports successfully")
        
        # Test argument parser creation
        parser = main.create_parser()
        print("  ✓ Argument parser created")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Main script test error: {e}")
        return False


def main():
    """Run all tests"""
    print("Multi-Camera Tracking System - Refactoring Verification")
    print("="*55)
    
    tests = [
        ("Import Tests", test_imports),
        ("Configuration System", test_config_system), 
        ("Database System", test_database),
        ("Directory Structure", test_directory_structure),
        ("Main Script", test_main_script)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                passed += 1
                print(f"  ✓ {test_name} PASSED")
            else:
                print(f"  ✗ {test_name} FAILED")
        except Exception as e:
            print(f"  ✗ {test_name} ERROR: {e}")
    
    print(f"\n{'='*55}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Refactoring successful.")
        return True
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
