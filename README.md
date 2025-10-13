# Multi-Camera Tracking System

A comprehensive multi-camera object tracking system that uses YOLO for object detection and advanced tracking algorithms to maintain consistent object identities across multiple camera views.

## Features

- **Multi-Camera Object Tracking**: Track objects across multiple camera views with consistent global IDs
- **YOLO Integration**: Uses YOLOv8/YOLO11 for robust object detection
- **Advanced Tracking**: Implements Kalman filtering and AMC-based motion prediction
- **Homography-Based View Fusion**: Combines multiple camera views into a unified coordinate system
- **Real-time Processing**: Supports both live tracking and batch processing modes
- **Database Integration**: Stores tracking data with SQLite for analysis and export
- **Configurable Parameters**: Comprehensive configuration system with YAML/JSON support
- **Modular Architecture**: Well-organized codebase with clear separation of concerns

## System Requirements

- Python 3.7+
- CUDA-capable GPU (recommended for YOLO inference)
- OpenCV 4.5+
- PyTorch 1.9+

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Multi-Camera-Tracking/intersection_method_1
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLO models:**
   ```bash
   # The system will automatically download models on first run
   # Or manually download:
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt
   ```

## Quick Start

### 1. Basic Tracking

Run multi-camera tracking with two video files:

```bash
python main.py track video1.mp4 video2.mp4 --mode live
```

### 2. Batch Processing

Process videos and save output:

```bash
python main.py track video1.mp4 video2.mp4 --mode batch --output result.mp4
```

### 3. Camera Calibration

Calculate homography matrix for camera alignment:

```bash
python main.py calibrate homography video1.mp4 video2.mp4
```

### 4. View Union

Create combined camera view:

```bash
python main.py union video1.mp4 video2.mp4
```

## Project Structure

```
intersection_method_1/
├── main.py                 # Main entry point
├── tracking_pipeline.py    # Enhanced tracking pipeline
├── run_pipeline.py        # View union pipeline
├── view_union.py          # View union implementation
├── multi_object_tracker.py # Multi-object tracking
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── config/               # Configuration files
├── data/                 # Data storage
├── logs/                 # Log files
├── src/                  # Source code modules
│   ├── detectors/        # Object detection modules
│   │   └── yolo_detector.py
│   ├── trackers/         # Tracking algorithms
│   │   ├── multi_camera_tracker.py
│   │   ├── kalman_tracker.py
│   │   └── yolo_multicam_tracker.py
│   ├── utils/            # Utility functions
│   │   ├── camera_utils.py
│   │   ├── homography_calculator.py
│   │   ├── point_selector.py
│   │   ├── database.py
│   │   └── config.py
│   └── calibration/      # Camera calibration
│       └── calibrate_cameras.py
└── models/               # YOLO model files
```

## Configuration

The system uses a hierarchical configuration system supporting YAML and JSON formats.

### Default Configuration

Create a configuration file `config/system_config.yaml`:

```yaml
detection:
  model_path: "yolo11n.pt"
  confidence_threshold: 0.6
  device: "auto"

tracking:
  distance_threshold: 50.0
  max_missing_frames: 30
  min_track_length: 5

camera:
  reference_camera: 2
  homography_path: "homography_matrix.npy"

database:
  db_path: "data/tracking_data.db"
  auto_cleanup_days: 30

logging:
  level: "INFO"
  file_path: "logs/tracking.log"
```

### Environment Variables

You can override configuration using environment variables:

```bash
export TRACKING_CONFIDENCE_THRESHOLD=0.7
export TRACKING_DISTANCE_THRESHOLD=75.0
export TRACKING_LOG_LEVEL=DEBUG
```

## Usage Examples

### Advanced Tracking with Custom Parameters

```bash
python main.py track video1.mp4 video2.mp4 \
  --distance-threshold 75 \
  --confidence-threshold 0.7 \
  --max-missing-frames 50 \
  --mode live
```

### Batch Processing with Configuration

```bash
python main.py track video1.mp4 video2.mp4 \
  --mode batch \
  --output enhanced_tracking.mp4 \
  --config config/custom_config.yaml
```

### System Information

```bash
python main.py info --check-deps
```

## Camera Setup and Calibration

### 1. Camera Calibration (Optional)

For accurate 3D positioning, calibrate your cameras:

```bash
python main.py calibrate camera --camera-id 1
python main.py calibrate camera --camera-id 2
```

### 2. Homography Calculation

Calculate transformation matrix between camera views:

```bash
python main.py calibrate homography video1.mp4 video2.mp4
```

This will:
- Open both videos side by side
- Allow you to select corresponding points
- Calculate and save the homography matrix

### 3. Point Selection Tips

When selecting matching points:
- Choose at least 4 corresponding points
- Select points on the ground plane for best results
- Avoid moving objects
- Distribute points across the overlapping area

## Database and Data Management

The system automatically stores tracking data in SQLite database:

### Database Schema

- **Sessions**: Tracking session metadata
- **Detections**: Individual object detections
- **Tracks**: Global track information

### Data Export

Export tracking data for analysis:

```python
from src.utils.database import TrackingDatabase

db = TrackingDatabase("data/tracking_data.db")
db.export_session_data("session_id", "export.json")
```

## Performance Optimization

### GPU Acceleration

Ensure CUDA is available for optimal performance:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

### Memory Management

For large videos, consider:
- Reducing input resolution
- Adjusting batch size
- Using frame skipping

### Configuration Tuning

Optimize parameters based on your use case:

- **High Accuracy**: Lower confidence threshold, higher distance threshold
- **Real-time Performance**: Higher confidence threshold, lower distance threshold
- **Dense Scenes**: Increase max_missing_frames
- **Sparse Scenes**: Decrease max_missing_frames

## Troubleshooting

### Common Issues

1. **YOLO Model Not Found**
   ```bash
   # Download manually
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt
   ```

2. **CUDA Out of Memory**
   ```bash
   # Use CPU instead
   python main.py track video1.mp4 video2.mp4 --device cpu
   ```

3. **Homography Calculation Fails**
   - Ensure sufficient overlap between camera views
   - Select points carefully on stable surfaces
   - Try selecting more than 4 points

4. **Poor Tracking Performance**
   - Adjust distance_threshold based on your scene
   - Tune confidence_threshold for your objects
   - Check camera calibration quality

### Debug Mode

Enable debug logging for troubleshooting:

```bash
python main.py track video1.mp4 video2.mp4 --log-level DEBUG
```

## API Reference

### Main Classes

#### YOLOv8Detector
```python
from src.detectors.yolo_detector import YOLOv8Detector

detector = YOLOv8Detector(model_path="yolo11n.pt", confidence_threshold=0.6)
detections = detector.detect(frame)
```

#### GlobalIDManager
```python
from src.trackers.multi_camera_tracker import GlobalIDManager

manager = GlobalIDManager(distance_threshold=50.0)
global_tracks = manager.process_detections(camera_detections)
```

#### TrackingDatabase
```python
from src.utils.database import TrackingDatabase

db = TrackingDatabase("tracking_data.db")
session_id = db.create_session(session_record)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Add docstrings to all functions
- Format code with Black

### Testing

Run tests before submitting:

```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLO implementation
- OpenCV community for computer vision tools
- PyTorch team for deep learning framework

## Support

For questions and support:

1. Check the troubleshooting section
2. Review existing issues
3. Create a new issue with detailed information

## Changelog

### Version 1.0.0
- Initial release
- Multi-camera tracking with global IDs
- YOLO integration
- Database support
- Configuration management
- Comprehensive documentation
