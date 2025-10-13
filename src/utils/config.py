#!/usr/bin/env python3
"""
Configuration Management Module

This module provides configuration management for the multi-camera tracking system,
including loading from files, environment variables, and command-line arguments.

Author: Multi-Camera Tracking System
"""

import json
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field
import logging


@dataclass
class DetectionConfig:
    """Configuration for object detection"""
    model_path: str = "yolo11n.pt"
    confidence_threshold: float = 0.6
    device: str = "auto"  # auto, cpu, cuda
    input_size: int = 640
    max_detections: int = 100


@dataclass
class TrackingConfig:
    """Configuration for object tracking"""
    distance_threshold: float = 50.0
    confidence_threshold: float = 0.6
    max_missing_frames: int = 30
    min_track_length: int = 5
    merge_iou_threshold: float = 0.3
    kalman_process_noise: float = 0.03
    kalman_measurement_noise: float = 0.1


@dataclass
class CameraConfig:
    """Configuration for camera setup"""
    camera1_id: int = 1
    camera2_id: int = 2
    reference_camera: int = 2
    homography_path: str = "homography_matrix.npy"
    calibration_path1: str = "cam1_calib.npz"
    calibration_path2: str = "cam2_calib.npz"


@dataclass
class DatabaseConfig:
    """Configuration for database"""
    db_path: str = "data/tracking_data.db"
    auto_cleanup_days: int = 30
    export_format: str = "json"  # json, csv
    backup_enabled: bool = True


@dataclass
class VideoConfig:
    """Configuration for video processing"""
    output_fps: float = 30.0
    output_codec: str = "mp4v"
    output_quality: int = 90
    resize_factor: float = 1.0
    frame_skip: int = 1


@dataclass
class UIConfig:
    """Configuration for user interface"""
    window_width: int = 1200
    window_height: int = 800
    show_fps: bool = True
    show_statistics: bool = True
    overlay_opacity: float = 0.7


@dataclass
class LoggingConfig:
    """Configuration for logging"""
    level: str = "INFO"
    file_path: str = "logs/tracking.log"
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class SystemConfig:
    """Main system configuration"""
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # System paths
    data_dir: str = "data"
    logs_dir: str = "logs"
    config_dir: str = "config"
    models_dir: str = "models"
    
    # Session settings
    session_timeout: int = 3600  # seconds
    auto_save_interval: int = 300  # seconds


class ConfigManager:
    """
    Configuration manager for the tracking system
    
    Handles loading, saving, and validation of configuration files
    in multiple formats (JSON, YAML).
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path (str, optional): Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config = SystemConfig()
        self.config_path = config_path
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            self.load_default_config()
    
    def load_default_config(self):
        """Load default configuration"""
        self.config = SystemConfig()
        self.logger.info("Loaded default configuration")
    
    def load_config(self, config_path: str):
        """
        Load configuration from file
        
        Args:
            config_path (str): Path to configuration file
        """
        try:
            config_path = Path(config_path)
            
            if not config_path.exists():
                self.logger.warning(f"Config file not found: {config_path}")
                self.load_default_config()
                return
            
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yml', '.yaml']:
                    data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {config_path.suffix}")
            
            # Update configuration with loaded data
            self._update_config_from_dict(data)
            self.config_path = str(config_path)
            
            self.logger.info(f"Loaded configuration from: {config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {e}")
            self.load_default_config()
    
    def _update_config_from_dict(self, data: Dict[str, Any]):
        """
        Update configuration from dictionary
        
        Args:
            data (dict): Configuration data
        """
        for section_name, section_data in data.items():
            if hasattr(self.config, section_name):
                section_config = getattr(self.config, section_name)
                
                if isinstance(section_config, (DetectionConfig, TrackingConfig, 
                                             CameraConfig, DatabaseConfig,
                                             VideoConfig, UIConfig, LoggingConfig)):
                    # Update dataclass fields
                    for key, value in section_data.items():
                        if hasattr(section_config, key):
                            setattr(section_config, key, value)
                else:
                    # Update simple attributes
                    setattr(self.config, section_name, section_data)
    
    def save_config(self, config_path: Optional[str] = None, format: str = "yaml"):
        """
        Save configuration to file
        
        Args:
            config_path (str, optional): Path to save configuration
            format (str): File format (yaml, json)
        """
        if config_path is None:
            config_path = self.config_path or f"config/system_config.{format}"
        
        try:
            config_path = Path(config_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert config to dictionary
            config_dict = asdict(self.config)
            
            with open(config_path, 'w') as f:
                if format.lower() in ['yml', 'yaml']:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif format.lower() == 'json':
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Saved configuration to: {config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save config to {config_path}: {e}")
    
    def update_from_args(self, args):
        """
        Update configuration from command-line arguments
        
        Args:
            args: Parsed command-line arguments
        """
        # Update detection config
        if hasattr(args, 'confidence_threshold'):
            self.config.detection.confidence_threshold = args.confidence_threshold
            self.config.tracking.confidence_threshold = args.confidence_threshold
        
        if hasattr(args, 'distance_threshold'):
            self.config.tracking.distance_threshold = args.distance_threshold
        
        if hasattr(args, 'max_missing_frames'):
            self.config.tracking.max_missing_frames = args.max_missing_frames
        
        if hasattr(args, 'min_track_length'):
            self.config.tracking.min_track_length = args.min_track_length
        
        # Update logging config
        if hasattr(args, 'log_level'):
            self.config.logging.level = args.log_level
        
        if hasattr(args, 'log_file'):
            if args.log_file:
                self.config.logging.file_path = args.log_file
        
        self.logger.info("Updated configuration from command-line arguments")
    
    def update_from_env(self):
        """Update configuration from environment variables"""
        env_mappings = {
            'TRACKING_CONFIDENCE_THRESHOLD': ('detection', 'confidence_threshold', float),
            'TRACKING_DISTANCE_THRESHOLD': ('tracking', 'distance_threshold', float),
            'TRACKING_DB_PATH': ('database', 'db_path', str),
            'TRACKING_LOG_LEVEL': ('logging', 'level', str),
            'TRACKING_MODEL_PATH': ('detection', 'model_path', str),
            'TRACKING_DEVICE': ('detection', 'device', str),
        }
        
        for env_var, (section, key, type_func) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    converted_value = type_func(value)
                    section_config = getattr(self.config, section)
                    setattr(section_config, key, converted_value)
                    self.logger.info(f"Updated {section}.{key} from environment: {converted_value}")
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Invalid environment value for {env_var}: {value}")
    
    def validate_config(self) -> bool:
        """
        Validate configuration settings
        
        Returns:
            bool: True if configuration is valid
        """
        errors = []
        
        # Validate detection config
        if not (0.0 <= self.config.detection.confidence_threshold <= 1.0):
            errors.append("Detection confidence threshold must be between 0.0 and 1.0")
        
        # Validate tracking config
        if self.config.tracking.distance_threshold <= 0:
            errors.append("Tracking distance threshold must be positive")
        
        if self.config.tracking.max_missing_frames <= 0:
            errors.append("Max missing frames must be positive")
        
        # Validate paths
        model_path = Path(self.config.detection.model_path)
        if not model_path.exists() and not model_path.name.startswith(('yolo', 'best')):
            errors.append(f"Model file not found: {model_path}")
        
        # Log errors
        for error in errors:
            self.logger.error(f"Configuration validation error: {error}")
        
        return len(errors) == 0
    
    def get_section(self, section_name: str):
        """
        Get configuration section
        
        Args:
            section_name (str): Name of configuration section
            
        Returns:
            Configuration section object
        """
        return getattr(self.config, section_name, None)
    
    def create_directories(self):
        """Create necessary directories based on configuration"""
        directories = [
            self.config.data_dir,
            self.config.logs_dir,
            self.config.config_dir,
            self.config.models_dir,
            Path(self.config.database.db_path).parent,
            Path(self.config.logging.file_path).parent
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Created necessary directories")


# Global configuration instance
config_manager = ConfigManager()


def get_config() -> SystemConfig:
    """Get the global configuration instance"""
    return config_manager.config


def load_config(config_path: str):
    """Load configuration from file"""
    global config_manager
    config_manager.load_config(config_path)


def save_config(config_path: str, format: str = "yaml"):
    """Save configuration to file"""
    config_manager.save_config(config_path, format)


# Example usage and testing
if __name__ == "__main__":
    # Create configuration manager
    manager = ConfigManager()
    
    # Print current configuration
    print("Default Configuration:")
    print(f"Detection confidence: {manager.config.detection.confidence_threshold}")
    print(f"Tracking distance: {manager.config.tracking.distance_threshold}")
    print(f"Database path: {manager.config.database.db_path}")
    
    # Save configuration
    manager.save_config("config/example_config.yaml")
    manager.save_config("config/example_config.json", format="json")
    
    # Validate configuration
    is_valid = manager.validate_config()
    print(f"Configuration valid: {is_valid}")
    
    # Create directories
    manager.create_directories()
