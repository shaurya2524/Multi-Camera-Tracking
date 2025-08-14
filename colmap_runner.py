import os
import subprocess
import sys
from pathlib import Path
import shutil

class COLMAPRunner:
    def __init__(self, project_path, colmap_exe=None):
        self.project_path = Path(project_path)
        self.colmap_exe = colmap_exe or "colmap"  # Assumes colmap is in PATH
        
        # Create directory structure
        self.database_path = self.project_path / "database.db"
        self.images_path = self.project_path / "images"
        self.sparse_path = self.project_path / "sparse"
        self.dense_path = self.project_path / "dense"
        
        # Create directories
        self.sparse_path.mkdir(exist_ok=True)
        self.dense_path.mkdir(exist_ok=True)
    
    def run_command(self, cmd):
        """Run a command and handle errors"""
        print(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("Command completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Command failed with error: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return False
    
    def extract_features(self, camera_model="OPENCV"):
        """Extract features from images"""
        cmd = [
            self.colmap_exe, "feature_extractor",
            "--database_path", str(self.database_path),
            "--image_path", str(self.images_path),
            "--ImageReader.camera_model", camera_model,
            "--ImageReader.single_camera", "0",  # Multiple cameras
            "--SiftExtraction.max_image_size", "1600",
            "--SiftExtraction.max_num_features", "8192"
        ]
        return self.run_command(cmd)
    
    def match_features(self, matching="exhaustive"):
        """Match features between images"""
        if matching == "exhaustive":
            cmd = [
                self.colmap_exe, "exhaustive_matcher",
                "--database_path", str(self.database_path),
                "--SiftMatching.guided_matching", "1",
                "--SiftMatching.max_ratio", "0.8",
                "--SiftMatching.max_distance", "0.7"
            ]
        else:  # sequential matching (faster but less accurate)
            cmd = [
                self.colmap_exe, "sequential_matcher",
                "--database_path", str(self.database_path),
                "--SequentialMatching.overlap", "10"
            ]
        return self.run_command(cmd)
    
    def run_mapper(self):
        """Run structure from motion"""
        cmd = [
            self.colmap_exe, "mapper",
            "--database_path", str(self.database_path),
            "--image_path", str(self.images_path),
            "--output_path", str(self.sparse_path),
            "--Mapper.ba_refine_focal_length", "1",
            "--Mapper.ba_refine_principal_point", "1",
            "--Mapper.ba_refine_extra_params", "1",
            "--Mapper.min_model_size", "10",
            "--Mapper.max_model_overlap", "20"
        ]
        return self.run_command(cmd)
    
    def undistort_images(self, model_path=None):
        """Undistort images for dense reconstruction"""
        if model_path is None:
            model_path = self.sparse_path / "0"
        
        cmd = [
            self.colmap_exe, "image_undistorter",
            "--image_path", str(self.images_path),
            "--input_path", str(model_path),
            "--output_path", str(self.dense_path),
            "--output_type", "COLMAP"
        ]
        return self.run_command(cmd)
    
    def patch_match_stereo(self):
        """Run patch match stereo"""
        cmd = [
            self.colmap_exe, "patch_match_stereo",
            "--workspace_path", str(self.dense_path),
            "--PatchMatchStereo.geom_consistency", "1",
            "--PatchMatchStereo.max_image_size", "1600"
        ]
        return self.run_command(cmd)
    
    def stereo_fusion(self):
        """Fuse stereo results into point cloud"""
        output_path = self.dense_path / "fused.ply"
        cmd = [
            self.colmap_exe, "stereo_fusion",
            "--workspace_path", str(self.dense_path),
            "--output_path", str(output_path),
            "--StereoFusion.min_num_pixels", "5",
            "--StereoFusion.max_num_pixels", "10000"
        ]
        
        success = self.run_command(cmd)
        if success:
            print(f"Point cloud saved to: {output_path}")
        return success
    
    def run_full_pipeline(self, skip_dense=False):
        """Run the complete COLMAP pipeline"""
        print("Starting COLMAP pipeline...")
        
        # Step 1: Feature extraction
        print("\n1. Extracting features...")
        if not self.extract_features():
            print("Feature extraction failed!")
            return False
        
        # Step 2: Feature matching
        print("\n2. Matching features...")
        if not self.match_features("exhaustive"):
            print("Feature matching failed!")
            return False
        
        # Step 3: Structure from Motion
        print("\n3. Running Structure from Motion...")
        if not self.run_mapper():
            print("Structure from Motion failed!")
            return False
        
        if skip_dense:
            print("Skipping dense reconstruction as requested")
            return True
        
        # Step 4: Dense reconstruction
        print("\n4. Undistorting images...")
        if not self.undistort_images():
            print("Image undistortion failed!")
            return False
        
        print("\n5. Running patch match stereo...")
        if not self.patch_match_stereo():
            print("Patch match stereo failed!")
            return False
        
        print("\n6. Fusing stereo results...")
        if not self.stereo_fusion():
            print("Stereo fusion failed!")
            return False
        
        print("\nCOLMAP pipeline completed successfully!")
        return True
    
    def export_to_other_formats(self):
        """Export results to other formats"""
        sparse_model = self.sparse_path / "0"
        if not sparse_model.exists():
            print("No sparse model found to export")
            return
        
        # Export to text format
        text_output = self.project_path / "sparse_text"
        text_output.mkdir(exist_ok=True)
        
        cmd = [
            self.colmap_exe, "model_converter",
            "--input_path", str(sparse_model),
            "--output_path", str(text_output),
            "--output_type", "TXT"
        ]
        
        if self.run_command(cmd):
            print(f"Sparse model exported to text format: {text_output}")
        
        # Export dense point cloud to PLY if it exists
        dense_ply = self.dense_path / "fused.ply"
        if dense_ply.exists():
            print(f"Dense point cloud available at: {dense_ply}")

# Usage example
if __name__ == "__main__":
    # Set your project path
    project_path = r"D:\Multi_Camera_Tracking\Wildtrack\colmap_input"
    
    # Initialize COLMAP runner
    # If colmap is not in PATH, provide full path to executable
    # colmap_exe = r"C:\path\to\colmap\COLMAP.exe"  # Windows
    colmap_runner = COLMAPRunner(project_path)
    
    # Check if images directory exists
    if not colmap_runner.images_path.exists():
        print(f"Images directory not found: {colmap_runner.images_path}")
        print("Please run the data preparation script first!")
        sys.exit(1)
    
    # Run the full pipeline
    # Set skip_dense=True if you only want sparse reconstruction (faster)
    success = colmap_runner.run_full_pipeline(skip_dense=False)
    
    if success:
        print("\nReconstruction completed!")
        colmap_runner.export_to_other_formats()
    else:
        print("\nReconstruction failed!")