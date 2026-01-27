# YOLO Video Processing Cheat Sheet

Quick reference for common video processing tasks.

---

## 🚀 Quick Start

### Basic Setup

```python
from app.services.ai.yolo_service import YOLOService
from app.services.ai.video_processor import VideoProcessor

# Initialize
yolo = YOLOService(model_path="./data/models/yolo/yolov8n.pt", device="cpu")
processor = VideoProcessor(yolo)
```

### GPU Setup

```python
# Use GPU for 10-100x speedup
yolo = YOLOService(model_path="./data/models/yolo/yolov8n.pt", device="cuda")
```

---

## 📹 Video Information

```python
# Get video metadata
info = processor.get_video_info("video.mp4")
print(f"Frames: {info['total_frames']}")
print(f"FPS: {info['fps']}")
print(f"Duration: {info['duration']}s")
print(f"Resolution: {info['width']}x{info['height']}")

# Validate video
is_valid, error = processor.validate_video("video.mp4")
if not is_valid:
    print(f"Error: {error}")
```

---

## 🎬 Processing Methods

### 1. Simple Detection (No Tracking)

```python
# Basic frame-by-frame detection
results = processor.process_video_simple(
    video_path="input.mp4",
    output_path="output.mp4",
    confidence=0.5,
    sample_rate=1  # 1=all frames, 5=every 5th frame
)
```

### 2. Object Tracking

```python
# Track objects with unique IDs across frames
results = processor.process_video_with_tracking(
    video_path="input.mp4",
    output_path="tracked.mp4",
    confidence=0.5
)

# Access tracked objects
for track_id, detections in results['tracks'].items():
    print(f"Track {track_id}: {len(detections)} frames")
```

### 3. Key Frame Extraction

```python
# Extract N evenly-spaced frames
frames = processor.extract_key_frames("video.mp4", num_frames=10)

for frame_num, frame in frames:
    detections = yolo.detect(frame, confidence=0.5)
    print(f"Frame {frame_num}: {len(detections)} objects")
```

### 4. Quick Preview

```python
# Fast preview (process every Nth frame)
results = processor.quick_preview(
    video_path="input.mp4",
    output_path="preview.mp4",
    sample_every=30,  # 1 frame per second at 30fps
    confidence=0.5
)
```

### 5. Summary Video

```python
# Create highlights with high-confidence detections only
results = processor.process_video_simple("input.mp4", "full.mp4")

summary = processor.create_summary_video(
    video_path="input.mp4",
    output_path="highlights.mp4",
    detections=results['detections'],
    confidence=0.7  # Only 70%+ confidence
)
```

---

## 📊 Results Analysis

### Detection Results

```python
results = processor.process_video_simple("video.mp4", "output.mp4")

print(f"Total frames: {results['total_frames']}")
print(f"Frames processed: {results['frames_processed']}")
print(f"Total detections: {results['total_detections']}")
print(f"Avg detections/frame: {results['avg_detections_per_frame']:.2f}")
print(f"Processing time: {results['processing_time_ms']/1000:.2f}s")
print(f"Avg time/frame: {results['avg_time_per_frame']:.2f}ms")

# Class distribution
class_counts = {}
for det in results['detections']:
    cls = det['class']
    class_counts[cls] = class_counts.get(cls, 0) + 1

for cls, count in class_counts.items():
    print(f"{cls}: {count}")
```

### Tracking Results

```python
results = processor.process_video_with_tracking("video.mp4", "tracked.mp4")

print(f"Unique tracks: {results['unique_tracks']}")
print(f"Total detections: {results['total_detections']}")

# Track durations
for track_id, duration in results['track_durations'].items():
    print(f"Track {track_id}: {duration} frames")

# Track details
for track_id, detections in results['tracks'].items():
    obj_class = detections[0]['class']
    avg_conf = sum(d['confidence'] for d in detections) / len(detections)
    print(f"Track {track_id}: {obj_class} (avg conf: {avg_conf:.2f})")
```

---

## 🎛️ Progress Monitoring

### Progress Callback

```python
def show_progress(frame_num, total_frames):
    progress = (frame_num / total_frames) * 100
    print(f"Progress: {frame_num}/{total_frames} ({progress:.1f}%)", end='\r')

results = processor.process_video_simple(
    video_path="input.mp4",
    output_path="output.mp4",
    progress_callback=show_progress
)
```

### Real-time Updates

```python
import time

def monitor_progress(frame_num, total_frames):
    if frame_num % 30 == 0:  # Update every 30 frames
        elapsed = time.time() - start_time
        fps = frame_num / elapsed if elapsed > 0 else 0
        eta = (total_frames - frame_num) / fps if fps > 0 else 0
        print(f"Frame {frame_num}/{total_frames} | {fps:.1f} FPS | ETA: {eta:.0f}s")

start_time = time.time()
results = processor.process_video_simple(
    video_path="input.mp4",
    output_path="output.mp4",
    progress_callback=monitor_progress
)
```

---

## ⚡ Performance Optimization

### Use GPU

```python
# CUDA (NVIDIA)
yolo = YOLOService(model_path="yolov8n.pt", device="cuda")

# MPS (Apple Silicon)
yolo = YOLOService(model_path="yolov8n.pt", device="mps")

# Check GPU availability
import torch
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Model Selection

```python
# Fastest (real-time capable)
yolo = YOLOService(model_path="yolov8n.pt")  # Nano

# Balanced (recommended)
yolo = YOLOService(model_path="yolov8s.pt")  # Small

# High accuracy
yolo = YOLOService(model_path="yolov8m.pt")  # Medium

# Best accuracy (slow)
yolo = YOLOService(model_path="yolov8x.pt")  # XLarge
```

### Sample Rate

```python
# All frames (slowest, most accurate)
results = processor.process_video_simple("video.mp4", "out.mp4", sample_rate=1)

# Every 5th frame (5x faster)
results = processor.process_video_simple("video.mp4", "out.mp4", sample_rate=5)

# Every 30th frame (~1 fps at 30fps, 30x faster)
results = processor.process_video_simple("video.mp4", "out.mp4", sample_rate=30)
```

### Resolution Scaling

```python
import cv2

def process_scaled(video_path, output_path, scale=0.5):
    """Process at lower resolution for speed"""
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Scale down
        height, width = frame.shape[:2]
        frame_scaled = cv2.resize(frame, (int(width*scale), int(height*scale)))
        
        # Detect on scaled frame
        detections = yolo.detect(frame_scaled, confidence=0.5)
        
        # Scale detections back up
        for det in detections:
            det.bbox = {k: v/scale for k, v in det.bbox.items()}
```

---

## 🎯 Common Use Cases

### Security Camera Analysis

```python
# Detect people and vehicles
results = processor.process_video_with_tracking(
    video_path="security_footage.mp4",
    output_path="analyzed.mp4",
    confidence=0.6
)

# Filter specific classes
people = [d for d in results['detections'] if d['class'] == 'person']
vehicles = [d for d in results['detections'] if d['class'] in ['car', 'truck', 'bus']]

print(f"People detected: {len(people)}")
print(f"Vehicles detected: {len(vehicles)}")
```

### Wildlife Monitoring

```python
# Focus on animal classes
animal_classes = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
                  'elephant', 'bear', 'zebra', 'giraffe']

results = processor.process_video_with_tracking("wildlife.mp4", "tracked.mp4")

animals = [d for d in results['detections'] 
           if d['class'] in animal_classes]

# Group by species
species_counts = {}
for det in animals:
    species = det['class']
    species_counts[species] = species_counts.get(species, 0) + 1

print("Species detected:")
for species, count in species_counts.items():
    print(f"  {species}: {count}")
```

### Traffic Analysis

```python
# Count vehicles by type
results = processor.process_video_with_tracking("traffic.mp4", "analyzed.mp4")

vehicle_types = {
    'car': 0,
    'truck': 0,
    'bus': 0,
    'motorcycle': 0
}

for det in results['detections']:
    if det['class'] in vehicle_types:
        vehicle_types[det['class']] += 1

print(f"Total vehicles: {sum(vehicle_types.values())}")
for vtype, count in vehicle_types.items():
    print(f"  {vtype}: {count}")
```

### Sports Analysis

```python
# Track players and ball
results = processor.process_video_with_tracking("game.mp4", "tracked.mp4")

# Find the ball (sports ball class)
ball_detections = [d for d in results['detections'] 
                   if d['class'] == 'sports ball']

# Track ball trajectory
ball_positions = [(d['frame'], d['bbox']['x1'], d['bbox']['y1']) 
                  for d in ball_detections]

print(f"Ball tracked in {len(ball_positions)} frames")
```

---

## 🔍 Frame-Level Operations

### Extract Specific Frames

```python
# Extract frames with detections
results = processor.process_video_simple("video.mp4", "output.mp4")

frames_with_objects = set(d['frame'] for d in results['detections'])

import cv2
cap = cv2.VideoCapture("video.mp4")

for frame_num in sorted(frames_with_objects):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(f"frame_{frame_num:05d}.jpg", frame)

cap.release()
```

### Process Frame Range

```python
# Process only frames 100-200
cap = cv2.VideoCapture("video.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)

for i in range(100, 200):
    ret, frame = cap.read()
    if not ret:
        break
    
    detections = yolo.detect(frame, confidence=0.5)
    annotated = yolo.draw_detections(frame, detections)
    # Process annotated frame

cap.release()
```

---

## 🐛 Error Handling

### Robust Processing

```python
def safe_process(video_path, output_path):
    try:
        # Validate first
        is_valid, error = processor.validate_video(video_path)
        if not is_valid:
            print(f"Invalid video: {error}")
            return None
        
        # Check video info
        info = processor.get_video_info(video_path)
        if info['duration'] > 600:  # 10 minutes
            print("Video too long, using preview mode")
            return processor.quick_preview(video_path, output_path, sample_every=30)
        
        # Process normally
        return processor.process_video_simple(video_path, output_path)
    
    except Exception as e:
        print(f"Error processing video: {e}")
        return None

result = safe_process("input.mp4", "output.mp4")
```

### Memory Management

```python
import psutil

def check_resources():
    """Check if enough resources available"""
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('.')
    
    if memory.percent > 90:
        raise RuntimeError(f"Low memory: {memory.percent}% used")
    
    if disk.percent > 95:
        raise RuntimeError(f"Low disk space: {disk.percent}% used")

# Check before processing
check_resources()
results = processor.process_video_simple("video.mp4", "output.mp4")
```

---

## 📈 Performance Benchmarks

### Timing Template

```python
import time

start = time.time()

results = processor.process_video_simple("video.mp4", "output.mp4")

elapsed = time.time() - start

print(f"\nPerformance:")
print(f"  Total time: {elapsed:.2f}s")
print(f"  Frames/sec: {results['total_frames']/elapsed:.2f}")
print(f"  Time/frame: {elapsed/results['total_frames']*1000:.2f}ms")
print(f"  Speedup needed for real-time: {results['total_frames']/(elapsed*info['fps']):.2f}x")
```

### Comparison Script

```python
def benchmark_models():
    models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']
    video = "test_video.mp4"
    
    for model_path in models:
        yolo = YOLOService(model_path=f"./data/models/yolo/{model_path}")
        processor = VideoProcessor(yolo)
        
        start = time.time()
        results = processor.quick_preview(video, f"output_{model_path}.mp4", sample_every=30)
        elapsed = time.time() - start
        
        print(f"{model_path}: {elapsed:.2f}s, {results['total_detections']} detections")
```

---

## 💾 File Management

### Clean Up Old Files

```python
from pathlib import Path
from datetime import datetime, timedelta

def cleanup_old_videos(directory, days=7):
    """Delete videos older than N days"""
    cutoff = datetime.now() - timedelta(days=days)
    
    for video_file in Path(directory).glob("*.mp4"):
        mod_time = datetime.fromtimestamp(video_file.stat().st_mtime)
        
        if mod_time < cutoff:
            print(f"Deleting old video: {video_file}")
            video_file.unlink()

cleanup_old_videos("./data/outputs", days=7)
```

### Batch Processing

```python
from pathlib import Path

def batch_process(input_dir, output_dir, **kwargs):
    """Process all videos in directory"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    video_files = list(Path(input_dir).glob("*.mp4"))
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing {video_path.name}")
        
        output_path = Path(output_dir) / f"processed_{video_path.name}"
        
        try:
            results = processor.process_video_simple(
                str(video_path),
                str(output_path),
                **kwargs
            )
            print(f"  ✅ Done: {results['total_detections']} detections")
        except Exception as e:
            print(f"  ❌ Failed: {e}")

batch_process("./input_videos", "./output_videos", confidence=0.5, sample_rate=5)
```

---

## 🔗 Quick Links

- **Full Tutorial**: `YOLO-Video-Processing.md`
- **Examples**: `video_detection_example.py`
- **README**: `README_VIDEO_DETECTION.md`
- **YOLO Docs**: https://docs.ultralytics.com/

---

**Quick Reference v1.0 | Video Processing with YOLO** 🎬