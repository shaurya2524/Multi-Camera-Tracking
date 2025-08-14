#!/usr/bin/env python3
"""
Complete multi-camera 3D reconstruction pipeline
Combines all the processing steps into a single workflow
"""

import os
import sys
import time
import shutil
import subprocess
from pathlib import Path
import argparse
import json
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import glob
from collections import defaultdict

class MultiCameraDataPreparator:
    """Handles data preparation and synchronization for multi-camera setup"""
    
    def __init__(self, base_path: Path, output_path: Path):
        self.base_path = Path(base_path)
        self.output_path = Path(output_path)
        self.cameras = []
        self.frame_counts = {}
        
    def discover_cameras(self) -> List[str]:
        """Discover available camera folders"""
        self.cameras = []
        for item in self.base_path.iterdir():
            if item.is_dir():
                # Check if it contains images
                image_files = list(item.glob("*.jpg")) + list(item.glob("*.png")) + list(item.glob("*.jpeg"))
                if image_files:
                    self.cameras.append(item.name)
                    self.frame_counts[item.name] = len(image_files)
        
        self.cameras.sort()
        return self.cameras
    
    def get_synchronized_frames(self, step: int = 1) -> Dict[str, List[Path]]:
        """Get synchronized frames from all cameras"""
        frames = defaultdict(list)
        
        # Find minimum frame count across all cameras
        min_frames = min(self.frame_counts.values()) if self.frame_counts else 0
        
        for camera in self.cameras:
            camera_path = self.base_path / camera
            image_files = sorted(list(camera_path.glob("*.jpg")) + 
                               list(camera_path.glob("*.png")) + 
                               list(camera_path.glob("*.jpeg")))
            
            # Take every 'step' frames up to min_frames
            selected_frames = image_files[::step][:min_frames//step]
            frames[camera] = selected_frames
        
        return frames
    
    def copy_synchronized_images(self, frames: Dict[str, List[Path]]) -> Path:
        """Copy synchronized images to output directory"""
        images_dir = self.output_path / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        for camera, frame_list in frames.items():
            camera_dir = images_dir / camera
            camera_dir.mkdir(exist_ok=True)
            
            for i, frame_path in enumerate(frame_list):
                dest_path = camera_dir / f"frame_{i:06d}{frame_path.suffix}"
                shutil.copy2(frame_path, dest_path)
        
        return images_dir
    
    def resize_images(self, images_dir: Path, target_size: Tuple[int, int]) -> Path:
        """Resize images to target size"""
        resized_dir = self.output_path / "images_resized"
        resized_dir.mkdir(parents=True, exist_ok=True)
        
        for camera in self.cameras:
            camera_input = images_dir / camera
            camera_output = resized_dir / camera
            camera_output.mkdir(exist_ok=True)
            
            for img_path in camera_input.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        resized = cv2.resize(img, target_size)
                        output_path = camera_output / img_path.name
                        cv2.imwrite(str(output_path), resized)
        
        return resized_dir

class BackgroundSubtractor:
    """Simple background subtraction for people removal"""
    
    def __init__(self):
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )
    
    def process_camera_folder(self, input_dir: Path, output_dir: Path) -> bool:
        """Process all images in a camera folder"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = sorted(list(input_dir.glob("*.jpg")) + 
                           list(input_dir.glob("*.png")) + 
                           list(input_dir.glob("*.jpeg")))
        
        if not image_files:
            return False
        
        # First pass: build background model
        for img_path in image_files[:min(100, len(image_files))]:  # Use first 100 frames
            img = cv2.imread(str(img_path))
            if img is not None:
                self.background_subtractor.apply(img)
        
        # Second pass: subtract background and save
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is not None:
                # Apply background subtraction
                fg_mask = self.background_subtractor.apply(img, learningRate=0)
                
                # Clean up the mask
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
                
                # Create background-only image
                bg_mask = cv2.bitwise_not(fg_mask)
                bg_img = cv2.bitwise_and(img, img, mask=bg_mask)
                
                # Fill foreground areas with background color or inpainting
                result = cv2.inpaint(img, fg_mask, 3, cv2.INPAINT_TELEA)
                
                output_path = output_dir / img_path.name
                cv2.imwrite(str(output_path), result)
        
        return True

class COLMAPRunner:
    """Handles COLMAP reconstruction pipeline"""
    
    def __init__(self, workspace_path: Path, colmap_exe: str = "colmap"):
        self.workspace_path = Path(workspace_path)
        self.colmap_exe = colmap_exe
        self.database_path = self.workspace_path / "database.db"
        self.sparse_path = self.workspace_path / "sparse"
        self.dense_path = self.workspace_path / "dense"
        
    def run_command(self, cmd: List[str]) -> bool:
        """Run a command and return success status"""
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {' '.join(cmd)}")
            print(f"Error: {e.stderr}")
            return False
    
    def feature_extraction(self, image_path: Path) -> bool:
        """Extract features from images"""
        cmd = [
            self.colmap_exe, "feature_extractor",
            "--database_path", str(self.database_path),
            "--image_path", str(image_path),
            "--ImageReader.single_camera", "0",
            "--SiftExtraction.use_gpu", "1"
        ]
        return self.run_command(cmd)
    
    def feature_matching(self) -> bool:
        """Match features between images"""
        cmd = [
            self.colmap_exe, "exhaustive_matcher",
            "--database_path", str(self.database_path),
            "--SiftMatching.use_gpu", "1"
        ]
        return self.run_command(cmd)
    
    def sparse_reconstruction(self, image_path: Path) -> bool:
        """Perform sparse reconstruction"""
        self.sparse_path.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            self.colmap_exe, "mapper",
            "--database_path", str(self.database_path),
            "--image_path", str(image_path),
            "--output_path", str(self.sparse_path)
        ]
        return self.run_command(cmd)
    
    def dense_reconstruction(self, image_path: Path) -> bool:
        """Perform dense reconstruction"""
        self.dense_path.mkdir(parents=True, exist_ok=True)
        
        # Find the reconstruction folder
        reconstruction_dirs = list(self.sparse_path.glob("*"))
        if not reconstruction_dirs:
            return False
        
        reconstruction_path = reconstruction_dirs[0]  # Use first/best reconstruction
        
        # Image undistortion
        cmd = [
            self.colmap_exe, "image_undistorter",
            "--image_path", str(image_path),
            "--input_path", str(reconstruction_path),
            "--output_path", str(self.dense_path),
            "--output_type", "COLMAP"
        ]
        if not self.run_command(cmd):
            return False
        
        # Patch match stereo
        cmd = [
            self.colmap_exe, "patch_match_stereo",
            "--workspace_path", str(self.dense_path),
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.geom_consistency", "1"
        ]
        if not self.run_command(cmd):
            return False
        
        # Stereo fusion
        cmd = [
            self.colmap_exe, "stereo_fusion",
            "--workspace_path", str(self.dense_path),
            "--workspace_format", "COLMAP",
            "--input_type", "geometric",
            "--output_path", str(self.dense_path / "fused.ply")
        ]
        return self.run_command(cmd)
    
    def run_full_pipeline(self, image_path: Path, skip_dense: bool = False) -> bool:
        """Run complete COLMAP pipeline"""
        # Remove old database
        if self.database_path.exists():
            self.database_path.unlink()
        
        # Feature extraction
        if not self.feature_extraction(image_path):
            return False
        
        # Feature matching
        if not self.feature_matching():
            return False
        
        # Sparse reconstruction
        if not self.sparse_reconstruction(image_path):
            return False
        
        # Dense reconstruction (optional)
        if not skip_dense:
            if not self.dense_reconstruction(image_path):
                return False
        
        return True

class PointCloudProcessor:
    """Process and clean point clouds"""
    
    def __init__(self, ply_path: Path):
        self.ply_path = Path(ply_path)
        self.points = None
        self.colors = None
        self.normals = None
        
    def load_ply(self) -> bool:
        """Load PLY file (basic implementation)"""
        try:
            # This is a simplified PLY reader - in practice you'd use Open3D or similar
            # For now, we'll just check if file exists
            return self.ply_path.exists()
        except Exception as e:
            print(f"Error loading PLY: {e}")
            return False
    
    def export_to_other_formats(self, output_dir: Path) -> bool:
        """Export to other formats"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy the original PLY file
        if self.ply_path.exists():
            shutil.copy2(self.ply_path, output_dir / "point_cloud.ply")
            return True
        
        return False

def create_summary_report(output_path: Path) -> bool:
    """Create a summary report of the reconstruction"""
    report_path = output_path / "summary_report.html"
    
    # Check what files exist
    sparse_exists = (output_path / "sparse").exists()
    dense_exists = (output_path / "dense" / "fused.ply").exists()
    images_exist = (output_path / "images").exists()
    
    html_content = f"""
    <html>
    <head><title>3D Reconstruction Report</title></head>
    <body>
        <h1>3D Reconstruction Summary</h1>
        <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Processing Status</h2>
        <ul>
            <li>Images prepared: {'✓' if images_exist else '✗'}</li>
            <li>Sparse reconstruction: {'✓' if sparse_exists else '✗'}</li>
            <li>Dense reconstruction: {'✓' if dense_exists else '✗'}</li>
        </ul>
        
        <h2>Output Files</h2>
        <ul>
            <li>Workspace: {output_path}</li>
            <li>Point cloud: {output_path / 'dense' / 'fused.ply' if dense_exists else 'Not generated'}</li>
        </ul>
    </body>
    </html>
    """
    
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    return True

class ReconstructionPipeline:
    def __init__(self, config):
        self.config = config
        self.base_path = Path(config['base_path'])
        self.output_path = Path(config['output_path'])
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Create log file
        self.log_file = self.output_path / "reconstruction_log.txt"
        
    def log(self, message):
        """Log message to both console and file"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def step1_prepare_data(self):
        """Step 1: Prepare and synchronize data"""
        self.log("=== Step 1: Data Preparation ===")
        
        try:
            prep = MultiCameraDataPreparator(self.base_path, self.output_path)
            
            # Discover cameras
            cameras = prep.discover_cameras()
            self.log(f"Found {len(cameras)} cameras: {cameras}")
            
            if not cameras:
                self.log("No cameras found!")
                return False
            
            # Get synchronized frames
            frames = prep.get_synchronized_frames(step=self.config['frame_step'])
            self.log(f"Selected {len(frames[cameras[0]]) if cameras else 0} frames per camera")
            
            # Copy synchronized images
            images_dir = prep.copy_synchronized_images(frames)
            self.log(f"Images copied to: {images_dir}")
            
            # Resize images if requested
            if self.config['resize_images']:
                resized_dir = prep.resize_images(images_dir, self.config['target_size'])
                self.log(f"Images resized to: {resized_dir}")
                self.images_path = resized_dir
            else:
                self.images_path = images_dir
            
            self.log("Data preparation completed successfully")
            return True
            
        except Exception as e:
            self.log(f"Error in data preparation: {e}")
            return False
    
    def step2_remove_people(self):
        """Step 2: Remove people from images (optional)"""
        if not self.config['remove_people']:
            self.log("Skipping people removal step")
            return True
        
        self.log("=== Step 2: People Removal ===")
        
        try:
            if self.config['people_removal_method'] == 'background_subtraction':
                self.log("Using background subtraction method...")
                
                cleaned_dir = self.output_path / "images_cleaned"
                cleaned_dir.mkdir(parents=True, exist_ok=True)
                
                # Process each camera
                for camera_dir in self.images_path.iterdir():
                    if camera_dir.is_dir():
                        self.log(f"Processing camera: {camera_dir.name}")
                        
                        bg_subtractor = BackgroundSubtractor()
                        output_camera_dir = cleaned_dir / camera_dir.name
                        
                        success = bg_subtractor.process_camera_folder(camera_dir, output_camera_dir)
                        if success:
                            self.log(f"Successfully processed {camera_dir.name}")
                        else:
                            self.log(f"Failed to process {camera_dir.name}")
                
                self.images_path = cleaned_dir
            
            self.log("People removal completed successfully")
            return True
            
        except Exception as e:
            self.log(f"Error in people removal: {e}")
            return False
    
    def step3_run_colmap(self):
        """Step 3: Run COLMAP reconstruction"""
        self.log("=== Step 3: COLMAP Reconstruction ===")
        
        try:
            # Check if COLMAP is available
            try:
                subprocess.run([self.config['colmap_exe'], "--help"], 
                             check=True, capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.log("COLMAP not found. Please install COLMAP and ensure it's in your PATH.")
                return False
            
            colmap_runner = COLMAPRunner(self.output_path, self.config['colmap_exe'])
            
            # Flatten images for COLMAP (all images in one directory)
            flat_images_dir = self.output_path / "images_flat"
            flat_images_dir.mkdir(parents=True, exist_ok=True)
            
            for camera_dir in self.images_path.iterdir():
                if camera_dir.is_dir():
                    for img_file in camera_dir.glob("*"):
                        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            # Rename to include camera name
                            new_name = f"{camera_dir.name}_{img_file.name}"
                            shutil.copy2(img_file, flat_images_dir / new_name)
            
            success = colmap_runner.run_full_pipeline(
                flat_images_dir, 
                skip_dense=self.config['skip_dense']
            )
            
            if success:
                self.log("COLMAP reconstruction completed successfully")
            else:
                self.log("COLMAP reconstruction failed")
            
            return success
            
        except Exception as e:
            self.log(f"Error in COLMAP reconstruction: {e}")
            return False
    
    def step4_process_pointcloud(self):
        """Step 4: Process and clean point cloud"""
        self.log("=== Step 4: Point Cloud Processing ===")
        
        try:
            dense_ply = self.output_path / "dense" / "fused.ply"
            
            if not dense_ply.exists():
                self.log("No dense point cloud found, skipping processing")
                return True
            
            processor = PointCloudProcessor(dense_ply)
            
            if processor.load_ply():
                output_dir = self.output_path / "processed"
                processor.export_to_other_formats(output_dir)
                self.log("Point cloud processing completed successfully")
            else:
                self.log("Failed to load point cloud")
                return False
            
            return True
            
        except Exception as e:
            self.log(f"Error in point cloud processing: {e}")
            return False
    
    def step5_generate_report(self):
        """Step 5: Generate summary report"""
        self.log("=== Step 5: Generating Report ===")
        
        try:
            create_summary_report(self.output_path)
            
            # Create a detailed report
            report = {
                'config': self.config,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'status': 'completed',
                'output_path': str(self.output_path)
            }
            
            report_file = self.output_path / "reconstruction_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.log(f"Report generated: {report_file}")
            return True
            
        except Exception as e:
            self.log(f"Error generating report: {e}")
            return False
    
    def run_pipeline(self):
        """Run the complete reconstruction pipeline"""
        self.log("Starting multi-camera 3D reconstruction pipeline...")
        
        start_time = time.time()
        
        steps = [
            ("Data Preparation", self.step1_prepare_data),
            ("People Removal", self.step2_remove_people),
            ("COLMAP Reconstruction", self.step3_run_colmap),
            ("Point Cloud Processing", self.step4_process_pointcloud),
            ("Report Generation", self.step5_generate_report)
        ]
        
        for step_name, step_func in steps:
            self.log(f"Starting: {step_name}")
            
            if not step_func():
                self.log(f"Pipeline failed at step: {step_name}")
                return False
        
        end_time = time.time()
        total_time = end_time - start_time
        
        self.log(f"Pipeline completed successfully in {total_time:.2f} seconds")
        return True

def create_config():
    """Create default configuration"""
    return {
        'base_path': r"D:\Multi_Camera_Tracking\Wildtrack\Image_subsets",
        'output_path': r"D:\Multi_Camera_Tracking\Wildtrack\reconstruction_output",
        'frame_step': 20,  # Use every 20th frame
        'resize_images': True,
        'target_size': (1280, 720),
        'remove_people': True,
        'people_removal_method': 'background_subtraction',  # or 'deep_learning'
        'colmap_exe': 'colmap',  # Path to COLMAP executable
        'skip_dense': False,  # Set to True for faster processing
        'voxel_size': 0.02,  # For point cloud downsampling
        'cameras': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Multi-camera 3D reconstruction pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--base-path', type=str, help='Base path to camera folders')
    parser.add_argument('--output-path', type=str, help='Output path for results')
    parser.add_argument('--skip-dense', action='store_true', help='Skip dense reconstruction')
    parser.add_argument('--no-people-removal', action='store_true', help='Skip people removal')
    parser.add_argument('--colmap-exe', type=str, help='Path to COLMAP executable')
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_config()
    
    # Override config with command line arguments
    if args.base_path:
        config['base_path'] = args.base_path
    if args.output_path:
        config['output_path'] = args.output_path
    if args.skip_dense:
        config['skip_dense'] = True
    if args.no_people_removal:
        config['remove_people'] = False
    if args.colmap_exe:
        config['colmap_exe'] = args.colmap_exe
    
    # Validate paths
    if not Path(config['base_path']).exists():
        print(f"Error: Base path does not exist: {config['base_path']}")
        return 1
    
    # Save configuration
    output_path = Path(config['output_path'])
    output_path.mkdir(parents=True, exist_ok=True)
    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to: {config_file}")
    print("Starting reconstruction pipeline...")
    
    # Run pipeline
    pipeline = ReconstructionPipeline(config)
    success = pipeline.run_pipeline()
    
    if success:
        print("Reconstruction completed successfully!")
        print(f"Results saved to: {config['output_path']}")
        return 0
    else:
        print("Reconstruction failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())