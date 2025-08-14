import cv2
import numpy as np
from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
import os

class PeopleRemover:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def download_deeplabv3(self):
        """Download DeepLabV3 model for segmentation"""
        try:
            import torchvision.models.segmentation as segmentation
            self.model = segmentation.deeplabv3_resnet101(pretrained=True)
            self.model.eval()
            self.model.to(self.device)
            print("DeepLabV3 model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def segment_people(self, image_path):
        """Segment people from image using DeepLabV3"""
        if not hasattr(self, 'model'):
            if not self.download_deeplabv3():
                return None
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        preprocess = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = preprocess(image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
            output_predictions = output.argmax(0).cpu().numpy()
        
        # Person class is 15 in COCO/Pascal VOC
        person_mask = (output_predictions == 15).astype(np.uint8)
        
        # Resize mask back to original size
        person_mask = cv2.resize(person_mask, original_size, interpolation=cv2.INTER_NEAREST)
        
        return person_mask
    
    def inpaint_people(self, image_path, output_path):
        """Remove people using inpainting"""
        # Load original image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not load image: {image_path}")
            return False
        
        # Get person mask
        person_mask = self.segment_people(image_path)
        if person_mask is None:
            print(f"Could not segment people in: {image_path}")
            return False
        
        # Dilate mask to ensure complete removal
        kernel = np.ones((5, 5), np.uint8)
        person_mask = cv2.dilate(person_mask, kernel, iterations=2)
        
        # Convert to 3-channel mask
        person_mask = (person_mask * 255).astype(np.uint8)
        
        # Inpaint to remove people
        inpainted = cv2.inpaint(image, person_mask, 3, cv2.INPAINT_TELEA)
        
        # Save result
        cv2.imwrite(str(output_path), inpainted)
        return True
    
    def process_directory(self, input_dir, output_dir):
        """Process all images in directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
        
        print(f"Processing {len(image_files)} images...")
        
        for i, img_file in enumerate(image_files):
            output_file = output_path / img_file.name
            
            if self.inpaint_people(img_file, output_file):
                print(f"Processed {i+1}/{len(image_files)}: {img_file.name}")
            else:
                print(f"Failed to process: {img_file.name}")
                # Copy original if processing fails
                import shutil
                shutil.copy2(img_file, output_file)

# Simpler background subtraction method (faster but less accurate)
class BackgroundSubtractor:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True)
    
    def process_camera_folder(self, camera_folder, output_folder):
        """Process all images from one camera to remove moving objects"""
        input_path = Path(camera_folder)
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        
        # Get all image files sorted by frame number
        image_files = sorted(input_path.glob('*.jpg'), 
                           key=lambda x: int(x.stem))
        
        print(f"Processing {len(image_files)} images from {camera_folder}")
        
        # Process images to build background model
        for i, img_file in enumerate(image_files):
            frame = cv2.imread(str(img_file))
            if frame is None:
                continue
                
            # Update background model
            fg_mask = self.bg_subtractor.apply(frame)
            
            # Every 10th frame, save the background
            if i % 10 == 0:
                background = self.bg_subtractor.getBackgroundImage()
                if background is not None:
                    output_file = output_path / f"background_{i:08d}.jpg"
                    cv2.imwrite(str(output_file), background)
        
        print(f"Background images saved to {output_path}")

# Usage example
if __name__ == "__main__":
    # Method 1: Deep learning based people removal (slower but more accurate)
    people_remover = PeopleRemover()
    
    input_dir = r"D:\Multi_Camera_Tracking\Wildtrack\colmap_input\images_resized"
    output_dir = r"D:\Multi_Camera_Tracking\Wildtrack\colmap_input\images_no_people"
    
    # Uncomment to use deep learning method
    # people_remover.process_directory(input_dir, output_dir)
    
    # Method 2: Background subtraction (faster, works well for static cameras)
    bg_subtractor = BackgroundSubtractor()
    
    base_path = r"D:\Multi_Camera_Tracking\Wildtrack\Image_subsets"
    output_base = r"D:\Multi_Camera_Tracking\Wildtrack\backgrounds"
    
    # Process each camera separately
    for camera in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']:
        camera_path = Path(base_path) / camera
        if camera_path.exists():
            output_path = Path(output_base) / camera
            bg_subtractor.process_camera_folder(camera_path, output_path)
    
    print("People removal preprocessing complete!")