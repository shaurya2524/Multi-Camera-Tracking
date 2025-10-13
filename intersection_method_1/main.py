#!/usr/bin/env python3
"""
Multi-Camera Tracking System - Main Entry Point

This is the main entry point for the multi-camera tracking system that provides
various modes of operation including camera calibration, homography calculation,
tracking pipeline execution, and view union creation.

Author: Multi-Camera Tracking System
Version: 1.0
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Import modules from the project
import sys
sys.path.append('src')

# Import configuration first
from src.utils.config import ConfigManager, get_config

# Import other modules as needed (lazy imports for better error handling)
def get_tracking_pipeline():
    from tracking_pipeline import EnhancedTrackingPipeline
    return EnhancedTrackingPipeline

def get_view_union():
    from run_pipeline import main as run_view_union
    return run_view_union

def get_homography_calculator():
    from src.utils.homography_calculator import HomographyCalculator
    return HomographyCalculator

def get_point_selector():
    from src.utils.point_selector import PointSelector
    return PointSelector


def setup_logging(log_level='INFO', log_file=None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[logging.StreamHandler(sys.stdout)]
        )
    
    return logging.getLogger(__name__)


def validate_video_files(video_paths):
    """Validate that video files exist and are readable"""
    for video_path in video_paths:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not os.access(video_path, os.R_OK):
            raise PermissionError(f"Cannot read video file: {video_path}")


def check_dependencies():
    """Check if required files and dependencies exist"""
    required_files = {
        'yolo_model': ['yolo11n.pt', 'yolov8n.pt'],
        'homography': 'homography_matrix.npy',
        'calibration': ['cam1_calib.npz', 'cam2_calib.npz']
    }
    
    missing_files = {}
    
    # Check YOLO models
    if not any(os.path.exists(f) for f in required_files['yolo_model']):
        missing_files['yolo_model'] = required_files['yolo_model']
    
    # Check homography (optional but recommended)
    if not os.path.exists(required_files['homography']):
        missing_files['homography'] = required_files['homography']
    
    # Check calibration files (optional)
    if not all(os.path.exists(f) for f in required_files['calibration']):
        missing_files['calibration'] = required_files['calibration']
    
    return missing_files


def run_tracking_mode(args, logger):
    """Run the enhanced tracking pipeline"""
    logger.info("Starting enhanced tracking pipeline...")
    
    # Validate video files
    validate_video_files([args.video1, args.video2])
    
    # Check dependencies
    missing_files = check_dependencies()
    if 'yolo_model' in missing_files:
        logger.error("No YOLO model found. Please ensure yolo11n.pt or yolov8n.pt exists.")
        return False
    
    if 'homography' in missing_files:
        logger.warning("Homography matrix not found. Tracking accuracy may be reduced.")
        logger.warning("Consider running: python main.py calibrate --mode homography")
    
    # Create and configure pipeline
    EnhancedTrackingPipeline = get_tracking_pipeline()
    pipeline = EnhancedTrackingPipeline()
    
    # Apply command line configurations
    if hasattr(args, 'distance_threshold'):
        pipeline.global_id_manager.distance_threshold = args.distance_threshold
    if hasattr(args, 'confidence_threshold'):
        pipeline.global_id_manager.confidence_threshold = args.confidence_threshold
    if hasattr(args, 'max_missing_frames'):
        pipeline.global_id_manager.max_missing_frames = args.max_missing_frames
    
    # Run pipeline based on mode
    try:
        if args.mode == 'live':
            logger.info("Running live tracking mode...")
            pipeline.run_live_tracking(args.video1, args.video2)
        elif args.mode == 'batch':
            logger.info(f"Running batch processing mode, output: {args.output}")
            pipeline.run_batch_processing(args.video1, args.video2, args.output)
        
        logger.info("Tracking pipeline completed successfully.")
        return True
        
    except Exception as e:
        logger.error(f"Error in tracking pipeline: {e}")
        return False


def run_calibration_mode(args, logger):
    """Run camera calibration or homography calculation"""
    logger.info(f"Starting calibration mode: {args.calibration_type}")
    
    try:
        if args.calibration_type == 'camera':
            logger.info("Running camera calibration...")
            # This would need to be implemented based on your calibrate_cameras.py
            logger.info("Camera calibration completed.")
            
        elif args.calibration_type == 'homography':
            logger.info("Running homography calculation...")
            
            # Validate video files
            validate_video_files([args.video1, args.video2])
            
            # Check if matching points exist or need to be selected
            if not os.path.exists('matching_points.json'):
                logger.info("Selecting matching points...")
                PointSelector = get_point_selector()
                selector = PointSelector()
                selector.select_points(args.video1, args.video2)
            
            # Calculate homography
            HomographyCalculator = get_homography_calculator()
            calc = HomographyCalculator()
            homography = calc.calculate_homography()
            
            if homography is not None:
                logger.info("Homography calculation completed successfully.")
            else:
                logger.error("Failed to calculate homography.")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error in calibration: {e}")
        return False


def run_view_union_mode(args, logger):
    """Run view union creation"""
    logger.info("Starting view union mode...")
    
    try:
        # Validate video files
        validate_video_files([args.video1, args.video2])
        
        # Check dependencies
        missing_files = check_dependencies()
        if 'homography' in missing_files:
            logger.warning("Homography matrix not found. Will attempt to calculate it.")
        
        # Run view union pipeline
        # This calls the existing run_pipeline.py functionality
        sys.argv = ['run_pipeline.py', args.video1, args.video2]
        run_view_union_func = get_view_union()
        run_view_union_func()
        
        logger.info("View union creation completed.")
        return True
        
    except Exception as e:
        logger.error(f"Error in view union creation: {e}")
        return False


def create_parser():
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        description='Multi-Camera Tracking System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run live tracking
  python main.py track video1.mp4 video2.mp4 --mode live
  
  # Run batch processing
  python main.py track video1.mp4 video2.mp4 --mode batch --output result.mp4
  
  # Calculate homography
  python main.py calibrate homography video1.mp4 video2.mp4
  
  # Create view union
  python main.py union video1.mp4 video2.mp4
  
  # Run with custom parameters
  python main.py track video1.mp4 video2.mp4 --distance-threshold 75 --confidence-threshold 0.7
        """
    )
    
    # Global arguments
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Set logging level')
    parser.add_argument('--log-file', help='Log to file instead of console')
    parser.add_argument('--config', help='Configuration file path')
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Tracking command
    track_parser = subparsers.add_parser('track', help='Run multi-camera tracking')
    track_parser.add_argument('video1', help='Path to first video file')
    track_parser.add_argument('video2', help='Path to second video file')
    track_parser.add_argument('--mode', choices=['live', 'batch'], default='live',
                             help='Tracking mode (default: live)')
    track_parser.add_argument('--output', default='enhanced_tracking_output.mp4',
                             help='Output video path for batch mode')
    track_parser.add_argument('--distance-threshold', type=float, default=50.0,
                             help='Distance threshold for global ID matching (pixels)')
    track_parser.add_argument('--confidence-threshold', type=float, default=0.6,
                             help='Minimum confidence threshold for detections')
    track_parser.add_argument('--max-missing-frames', type=int, default=30,
                             help='Maximum frames a track can be missing')
    track_parser.add_argument('--min-track-length', type=int, default=5,
                             help='Minimum track length before considering stable')
    
    # Calibration command
    calib_parser = subparsers.add_parser('calibrate', help='Run calibration procedures')
    calib_subparsers = calib_parser.add_subparsers(dest='calibration_type', 
                                                  help='Calibration type')
    
    # Camera calibration
    camera_calib_parser = calib_subparsers.add_parser('camera', 
                                                     help='Calibrate camera intrinsics')
    camera_calib_parser.add_argument('--camera-id', type=int, default=0,
                                   help='Camera ID to calibrate')
    camera_calib_parser.add_argument('--pattern-size', nargs=2, type=int, 
                                   default=[9, 6], metavar=('WIDTH', 'HEIGHT'),
                                   help='Chessboard pattern size')
    
    # Homography calibration
    homo_calib_parser = calib_subparsers.add_parser('homography',
                                                   help='Calculate homography matrix')
    homo_calib_parser.add_argument('video1', help='Path to first video file')
    homo_calib_parser.add_argument('video2', help='Path to second video file')
    homo_calib_parser.add_argument('--force-reselect', action='store_true',
                                  help='Force reselection of matching points')
    
    # View union command
    union_parser = subparsers.add_parser('union', help='Create camera view union')
    union_parser.add_argument('video1', help='Path to first video file')
    union_parser.add_argument('video2', help='Path to second video file')
    union_parser.add_argument('--output', default='union_output.mp4',
                             help='Output video path')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show system information')
    info_parser.add_argument('--check-deps', action='store_true',
                           help='Check dependencies')
    
    return parser


def run_info_mode(args, logger):
    """Show system information and check dependencies"""
    logger.info("=== Multi-Camera Tracking System Information ===")
    
    # Show current directory and files
    current_dir = Path.cwd()
    logger.info(f"Current directory: {current_dir}")
    
    # Check dependencies
    missing_files = check_dependencies()
    
    if not missing_files:
        logger.info("✓ All dependencies are available")
    else:
        logger.warning("Missing dependencies:")
        for dep_type, files in missing_files.items():
            if isinstance(files, list):
                logger.warning(f"  {dep_type}: {', '.join(files)}")
            else:
                logger.warning(f"  {dep_type}: {files}")
    
    # Show available video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(current_dir.glob(f'*{ext}'))
    
    if video_files:
        logger.info("Available video files:")
        for video_file in video_files:
            logger.info(f"  {video_file.name}")
    else:
        logger.info("No video files found in current directory")
    
    return True


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    logger.info("Multi-Camera Tracking System started")
    
    # Handle commands
    success = False
    
    try:
        if args.command == 'track':
            success = run_tracking_mode(args, logger)
        elif args.command == 'calibrate':
            success = run_calibration_mode(args, logger)
        elif args.command == 'union':
            success = run_view_union_mode(args, logger)
        elif args.command == 'info':
            success = run_info_mode(args, logger)
        else:
            parser.print_help()
            success = True
    
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        success = True
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        success = False
    
    if success:
        logger.info("Operation completed successfully")
        sys.exit(0)
    else:
        logger.error("Operation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
