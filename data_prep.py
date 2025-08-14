import os
import shutil
import cv2
import numpy as np
from pathlib import Path
import glob

class MultiCameraDataPreparator:
    def __init__(self, base_path, output_path):
        self.base_path = Path(base_path)
        self.output_path = Path(output_path)
        self.cameras = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
        
    def get_synchronized_frames(self, step=20):
        """Get frame numbers that exist across all cameras"""
        all_frames = {}
        
        # Get all frame numbers for each camera
        for camera in self.cameras:
            camera_path = self.base_path / camera
            if not camera_path.exists():
                print(f"Warning: Camera path {camera_path} does not exist")
                continue
                
            frames = []
            for img_file in camera_path.glob('*.jpg'):
                try:
                    frame_num = int(img_file.stem)
                    frames.append(frame_num)
                except:
                    continue
            all_frames[camera] = set(frames)
        
        # Find intersection of all frames
        common_frames = set.intersection(*all_frames.values()) if all_frames else set()
        
        # Sample frames with step
        sampled_frames = sorted(list(common_frames))[::step]
        
        print(f"Found {len(common_frames)} common frames across all cameras")
        print(f"Sampling every {step}th frame: {len(sampled_frames)} frames selected")
        
        return sampled_frames
    
    def copy_synchronized_images(self, frames_to_use=None, step=20):
        """Copy synchronized images to output directory"""
        if frames_to_use is None:
            frames_to_use = self.get_synchronized_frames(step)
        
        # Create output directory
        images_output = self.output_path / 'images'
        images_output.mkdir(parents=True, exist_ok=True)
        
        copied_count = 0
        for camera in self.cameras:
            camera_path = self.base_path / camera
            if not camera_path.exists():
                continue
                
            for frame_num in frames_to_use:
                src_file = camera_path / f"{frame_num:08d}.jpg"
                if src_file.exists():
                    # Name format: camera_framenumber.jpg
                    dst_file = images_output / f"{camera}_{frame_num:08d}.jpg"
                    shutil.copy2(src_file, dst_file)
                    copied_count += 1
        
        print(f"Copied {copied_count} images to {images_output}")
        return images_output
    
    def resize_images(self, target_size=(1920, 1080), quality=95):
        """Resize images to reduce processing time"""
        images_path = self.output_path / 'images'
        resized_path = self.output_path / 'images_resized'
        resized_path.mkdir(exist_ok=True)
        
        for img_file in images_path.glob('*.jpg'):
            img = cv2.imread(str(img_file))
            if img is not None:
                # Resize maintaining aspect ratio
                h, w = img.shape[:2]
                scale = min(target_size[0]/w, target_size[1]/h)
                new_w, new_h = int(w*scale), int(h*scale)
                
                resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Save with high quality
                cv2.imwrite(str(resized_path / img_file.name), resized, 
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
        
        print(f"Resized images saved to {resized_path}")
        return resized_path

# Usage example
if __name__ == "__main__":
    # Set your paths
    base_path = r"D:\Multi_Camera_Tracking\Wildtrack\Image_subsets"
    output_path = r"D:\Multi_Camera_Tracking\Wildtrack\colmap_input"
    
    # Initialize preprocessor
    prep = MultiCameraDataPreparator(base_path, output_path)
    
    # Get synchronized frames (every 20th frame to reduce data)
    frames = prep.get_synchronized_frames(step=20)
    
    # Copy synchronized images
    images_dir = prep.copy_synchronized_images(frames)
    
    # Optionally resize images to speed up processing
    resized_dir = prep.resize_images(target_size=(1280, 720))
    
    print("Data preparation complete!")
    print(f"Use images from: {resized_dir}")