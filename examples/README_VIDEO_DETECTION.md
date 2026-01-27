# YOLO Video Processing Tutorial

Complete guide for adding video processing capabilities to your YOLO object detection system.

## 📚 Overview

This tutorial extends the basic YOLO detection system with advanced video processing features:

- **Frame-by-frame detection** - Process video frames individually
- **Object tracking** - Track objects across frames with unique IDs
- **Video annotation** - Generate videos with bounding boxes and labels
- **Progress monitoring** - Real-time processing status updates
- **Video analytics** - Detailed statistics about detected objects

## 🎯 Quick Start

### 1. Prerequisites

Make sure you have completed the basic [YOLO Detection Tutorial](../../fastapi-devbox-boilerplate.wiki/YOLO-Detection.md) first.

Required dependencies:
```bash
poetry add opencv-python opencv-contrib-python ultralytics
```

### 2. Download YOLO Model

```bash
# Create model directory
mkdir -p data/models/yolo

# Download YOLOv8 nano model (smallest, fastest)
poetry run python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').save('data/models/yolo/yolov8n.pt')"
```

### 3. Run Examples

```bash
# Run all examples with your video
python examples/video_detection_example.py --video path/to/your/video.mp4

# Run specific example
python examples/video_detection_example.py --video test.mp4 --example tracking

# Specify output directory
python examples/video_detection_example.py --video test.mp4 --output ./results
```

## 📖 Documentation

Full tutorial available at: [`YOLO-Video-Processing.md`](../../fastapi-devbox-boilerplate.wiki/YOLO-Video-Processing.md)

The tutorial covers:
1. **Installation** - Dependencies and setup
2. **Database Models** - Video-specific fields
3. **Video Processor Service** - Core video processing logic
4. **Detection Service** - Job management
5. **API Endpoints** - REST API for video upload/processing
6. **Testing** - Comprehensive test examples
7. **Advanced Features** - WebSockets, summarization, optimization
8. **Troubleshooting** - Common issues and solutions

## 🎬 Example Usage

### Example 1: Simple Detection (No Tracking)

Best for: Static cameras, non-overlapping objects, quick analysis

```python
from app.services.ai.yolo_service import YOLOService
from app.services.ai.video_processor import VideoProcessor

# Initialize services
yolo_service = YOLOService(model_path="./data/models/yolo/yolov8n.pt")
video_processor = VideoProcessor(yolo_service)

# Process video
results = video_processor.process_video_simple(
    video_path="input.mp4",
    output_path="output_simple.mp4",
    confidence=0.5,
    sample_rate=1  # Process every frame
)

print(f"Detected {results['total_detections']} objects")
print(f"Processed in {results['processing_time_ms']/1000:.2f}s")
```

### Example 2: Object Tracking

Best for: Moving objects, crowded scenes, temporal analysis

```python
# Process with tracking
results = video_processor.process_video_with_tracking(
    video_path="input.mp4",
    output_path="output_tracked.mp4",
    confidence=0.5
)

print(f"Tracked {results['unique_tracks']} unique objects")
print(f"Total detections: {results['total_detections']}")

# Analyze specific tracks
for track_id, detections in results['tracks'].items():
    duration = len(detections)
    obj_class = detections[0]['class']
    print(f"Track {track_id}: {obj_class} appeared in {duration} frames")
```

### Example 3: Key Frame Extraction

Best for: Quick preview, thumbnails, summary generation

```python
# Extract 10 evenly-spaced frames
key_frames = video_processor.extract_key_frames(
    video_path="input.mp4",
    num_frames=10
)

# Process each frame
for frame_num, frame in key_frames:
    detections = yolo_service.detect(frame, confidence=0.5)
    annotated = yolo_service.draw_detections(frame, detections)
    # Save or display frame
```

### Example 4: Quick Preview

Best for: Long videos, fast analysis, testing

```python
# Process every 30th frame (1 per second at 30fps)
results = video_processor.quick_preview(
    video_path="input.mp4",
    output_path="preview.mp4",
    sample_every=30,
    confidence=0.5
)

print(f"Preview created in {results['processing_time_ms']/1000:.2f}s")
```

## 🔧 Advanced Features

### Progress Callbacks

Monitor processing progress in real-time:

```python
def show_progress(frame_num, total_frames):
    progress = (frame_num / total_frames) * 100
    print(f"Progress: {progress:.1f}%", end='\r')

results = video_processor.process_video_simple(
    video_path="input.mp4",
    output_path="output.mp4",
    progress_callback=show_progress
)
```

### Video Validation

Always validate videos before processing:

```python
is_valid, error_msg = video_processor.validate_video("input.mp4")

if not is_valid:
    print(f"Invalid video: {error_msg}")
else:
    # Proceed with processing
    pass
```

### Video Information

Get detailed video metadata:

```python
info = video_processor.get_video_info("input.mp4")
print(f"Resolution: {info['width']}x{info['height']}")
print(f"FPS: {info['fps']:.2f}")
print(f"Duration: {info['duration']:.2f}s")
print(f"Total frames: {info['total_frames']}")
```

### Summary Video

Create highlights with only high-confidence detections:

```python
# First, process the full video
results = video_processor.process_video_simple(
    video_path="input.mp4",
    output_path="full.mp4"
)

# Create summary with only confident detections
summary_path = video_processor.create_summary_video(
    video_path="input.mp4",
    output_path="highlights.mp4",
    detections=results['detections'],
    confidence=0.7  # Only frames with 70%+ confidence
)
```

## 🚀 Performance Tips

### 1. Use GPU Acceleration

```python
# Enable GPU for 10-100x speedup
yolo_service = YOLOService(
    model_path="./data/models/yolo/yolov8n.pt",
    device="cuda"  # or "mps" for Mac M1/M2
)
```

### 2. Choose the Right Model

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| yolov8n | Fastest | Good | Real-time, previews |
| yolov8s | Fast | Better | General purpose |
| yolov8m | Medium | Very Good | Balanced |
| yolov8l | Slow | Excellent | Accuracy critical |
| yolov8x | Slowest | Best | Maximum accuracy |

### 3. Sample Rate Optimization

```python
# Process every Nth frame for faster processing
results = video_processor.process_video_simple(
    video_path="input.mp4",
    output_path="output.mp4",
    sample_rate=5  # Process every 5th frame = 5x faster
)
```

### 4. Resolution Scaling

```python
import cv2

# Resize frame before detection
def preprocess_frame(frame):
    # Scale down to 640x480 for faster processing
    return cv2.resize(frame, (640, 480))
```

## 📊 Performance Benchmarks

Processing a 1-minute video (1920x1080, 30fps = 1800 frames):

| Hardware | Model | Sample Rate | Time | FPS Achieved |
|----------|-------|-------------|------|--------------|
| CPU (i7) | YOLOv8n | All frames | ~6 min | 5 fps |
| CPU (i7) | YOLOv8n | Every 5th | ~1.2 min | 25 fps |
| GPU (RTX 3060) | YOLOv8n | All frames | ~30 sec | 60 fps |
| GPU (RTX 3060) | YOLOv8m | All frames | ~1 min | 30 fps |
| GPU (A100) | YOLOv8x | All frames | ~20 sec | 90 fps |

## 🐛 Troubleshooting

### "Could not open video file"

**Cause:** Video codec not supported or file corrupted

**Solution:**
```bash
# Convert video to standard format
ffmpeg -i input.mov -c:v libx264 -c:a aac output.mp4
```

### "Codec not supported" when writing

**Cause:** OpenCV can't write with the specified codec

**Solution:** The `VideoProcessor` automatically tries multiple codecs. If issues persist:
```bash
# Install additional codecs
sudo apt-get install ffmpeg libavcodec-extra
```

### Out of memory with large videos

**Cause:** Video too large to process at once

**Solutions:**
1. Use sample rate: `sample_rate=5`
2. Process in smaller chunks
3. Use smaller model: `yolov8n.pt`
4. Reduce resolution before processing

### Slow processing on CPU

**Solutions:**
1. Use GPU: `device="cuda"`
2. Use smaller model: `yolov8n.pt`
3. Increase sample rate: `sample_rate=10`
4. Process preview first: `quick_preview()`

## 📁 Project Structure

```
apiboilerplate/
├── app/
│   ├── services/
│   │   └── ai/
│   │       ├── yolo_service.py        # YOLO detection service
│   │       └── video_processor.py      # Video processing service
│   ├── models/
│   │   └── detection.py                # Database models
│   ├── api/
│   │   └── video_detection.py          # API endpoints
│   └── schemas/
│       └── detection.py                # Response schemas
├── examples/
│   ├── video_detection_example.py      # Example scripts
│   └── README_VIDEO_DETECTION.md       # This file
├── data/
│   ├── models/yolo/                    # YOLO model files
│   ├── uploads/videos/                 # Uploaded videos
│   └── outputs/                        # Processed videos
└── fastapi-devbox-boilerplate.wiki/
    └── YOLO-Video-Processing.md        # Full tutorial
```

## 🎓 Learning Path

1. ✅ **Start here** - Run the example scripts
2. 📖 **Read tutorial** - [`YOLO-Video-Processing.md`](../../fastapi-devbox-boilerplate.wiki/YOLO-Video-Processing.md)
3. 🔧 **Implement API** - Add REST endpoints for video upload
4. 🚀 **Add features** - WebSockets, batch processing, etc.
5. 📦 **Deploy** - Production deployment with Docker

## 🔗 Related Resources

- **Main Tutorial**: [`YOLO-Video-Processing.md`](../../fastapi-devbox-boilerplate.wiki/YOLO-Video-Processing.md)
- **YOLO Detection**: [`YOLO-Detection.md`](../../fastapi-devbox-boilerplate.wiki/YOLO-Detection.md)
- **Ultralytics Docs**: https://docs.ultralytics.com/
- **OpenCV Video**: https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html

## 💡 Next Steps

After mastering video processing, try:

1. **Real-time Streaming** - Process webcam or RTSP streams
2. **Custom Training** - Train YOLO on your own dataset
3. **Mobile Deployment** - Export to ONNX/TFLite for mobile
4. **Multi-camera** - Process multiple video streams simultaneously
5. **Cloud Processing** - Scale with cloud GPUs (AWS, GCP)

## 📄 License

This tutorial is part of the FastAPI DevBox Boilerplate project.

---

## 🙋 Getting Help

- **Issues**: Open an issue in the main repository
- **Questions**: Check the wiki documentation
- **Discussions**: Join the community discussions

---

**Happy video processing! 🎬🚀**