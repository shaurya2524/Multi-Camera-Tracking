import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

class PersonReIDFeatureExtractor:
    """Feature extractor for person re-identification"""
    
    def __init__(self, model_type='resnet50', input_size=(256, 128)):
        """
        Initialize feature extractor
        
        Args:
            model_type: Type of backbone network
            input_size: Input image size (height, width)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[DEBUG] PersonReIDFeatureExtractor using device: {self.device}")
        self.input_size = input_size
        self.model_type = model_type
        
        # Initialize model
        self.model = self._build_model()
        self.model.eval()
        
        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def _build_model(self):
        """Build the feature extraction model"""
        if self.model_type == 'resnet50':
            # Use ResNet50 as backbone
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
            
            # Remove the final classification layer
            model = nn.Sequential(*list(model.children())[:-1])
            
            # Add global average pooling and feature reduction
            model.add_module('global_avg_pool', nn.AdaptiveAvgPool2d(1))
            model.add_module('flatten', nn.Flatten())
            model.add_module('feature_reduce', nn.Linear(2048, 512))
            model.add_module('bn', nn.BatchNorm1d(512))
            
        elif self.model_type == 'simple_cnn':
            # Simple CNN for basic feature extraction
            model = SimpleCNN()
        
        return model.to(self.device)
    
    def extract_features(self, person_crops: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract features from person crops
        
        Args:
            person_crops: List of person image crops
            
        Returns:
            List of feature vectors
        """
        if not person_crops:
            return []
        
        features = []
        
        with torch.no_grad():
            for crop in person_crops:
                if crop is None or crop.size == 0:
                    features.append(None)
                    continue
                
                # Preprocess crop
                try:
                    # Convert BGR to RGB
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    
                    # Apply transforms
                    input_tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
                    
                    # Extract features
                    feature_vector = self.model(input_tensor)
                    
                    # Normalize features
                    feature_vector = F.normalize(feature_vector, p=2, dim=1)
                    
                    # Convert to numpy
                    feature_np = feature_vector.cpu().numpy().flatten()
                    features.append(feature_np)
                    
                except Exception as e:
                    print(f"Error extracting features: {e}")
                    features.append(None)
        
        return features
    
    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Compute cosine similarity between two feature vectors
        
        Args:
            feat1: First feature vector
            feat2: Second feature vector
            
        Returns:
            Cosine similarity score
        """
        if feat1 is None or feat2 is None:
            return 0.0
        
        # Normalize features
        feat1_norm = feat1 / (np.linalg.norm(feat1) + 1e-8)
        feat2_norm = feat2 / (np.linalg.norm(feat2) + 1e-8)
        
        return np.dot(feat1_norm, feat2_norm)
    
    def extract_batch_features(self, person_crops: List[np.ndarray], 
                             batch_size: int = 32) -> List[np.ndarray]:
        """
        Extract features in batches for efficiency
        
        Args:
            person_crops: List of person image crops
            batch_size: Batch size for processing
            
        Returns:
            List of feature vectors
        """
        features = []
        
        for i in range(0, len(person_crops), batch_size):
            batch_crops = person_crops[i:i+batch_size]
            batch_features = self.extract_features(batch_crops)
            features.extend(batch_features)
        
        return features

class SimpleCNN(nn.Module):
    """Simple CNN for basic feature extraction"""
    
    def __init__(self, feature_dim=512):
        super(SimpleCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth conv block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class ColorHistogramExtractor:
    """Simple color histogram feature extractor as backup"""
    
    def __init__(self, bins=64):
        self.bins = bins
    
    def extract_features(self, person_crops: List[np.ndarray]) -> List[np.ndarray]:
        """Extract color histogram features"""
        features = []
        
        for crop in person_crops:
            if crop is None or crop.size == 0:
                features.append(None)
                continue
            
            # Convert to HSV
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            
            # Compute histograms for each channel
            hist_h = cv2.calcHist([hsv], [0], None, [self.bins], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [self.bins], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [self.bins], [0, 256])
            
            # Normalize histograms
            hist_h = hist_h.flatten() / (hist_h.sum() + 1e-8)
            hist_s = hist_s.flatten() / (hist_s.sum() + 1e-8)
            hist_v = hist_v.flatten() / (hist_v.sum() + 1e-8)
            
            # Concatenate histograms
            feature_vector = np.concatenate([hist_h, hist_s, hist_v])
            features.append(feature_vector)
        
        return features

class FeatureExtractorFactory:
    """Factory for creating different types of feature extractors"""
    
    @staticmethod
    def create_extractor(extractor_type='resnet50'):
        """Create feature extractor based on type"""
        if extractor_type == 'resnet50':
            return PersonReIDFeatureExtractor(model_type='resnet50')
        elif extractor_type == 'simple_cnn':
            return PersonReIDFeatureExtractor(model_type='simple_cnn')
        elif extractor_type == 'color_histogram':
            return ColorHistogramExtractor()
        else:
            raise ValueError(f"Unknown extractor type: {extractor_type}")