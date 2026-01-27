"""
YOLO Video Detection Example

This example demonstrates how to use the video processing service
to detect and track objects in videos.

Requirements:
    - YOLO model installed (yolov8n.pt)
    - OpenCV with video support
    - Test video file

Usage:
    python examples/video_detection_example.py --video path/to/video.mp4
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ai.yolo_service import YOLOService

from app.services.ai.video_processor import VideoProcessor


def process_video_simple_example(video_path: str, output_dir: str = "./output"):
    """
    Example 1: Simple frame-by-frame detection without tracking

    Best for: Quick analysis, static cameras, non-overlapping objects
    """
    print("\n" + "=" * 60)
    print("Example 1: Simple Detection (No Tracking)")
    print("=" * 60)

    # Initialize services
    print("🔧 Initializing YOLO service...")
    yolo_service = YOLOService(
        model_path="./data/models/yolo/yolov8n.pt",
        device="cpu",  # Change to "cuda" if you have GPU
    )

    video_processor = VideoProcessor(yolo_service)

    # Validate video
    print(f"📹 Validating video: {video_path}")
    is_valid, error_msg = video_processor.validate_video(video_path)
    if not is_valid:
        print(f"❌ Invalid video: {error_msg}")
        return

    # Get video info
    video_info = video_processor.get_video_info(video_path)
    print(f"✅ Video Info:")
    print(f"   - Resolution: {video_info['width']}x{video_info['height']}")
    print(f"   - FPS: {video_info['fps']:.2f}")
    print(f"   - Total frames: {video_info['total_frames']}")
    print(f"   - Duration: {video_info['duration']:.2f}s")

    # Process video
    output_path = Path(output_dir) / f"{Path(video_path).stem}_simple.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n⏳ Processing video (this may take a while)...")
    print(f"   Output will be saved to: {output_path}")

    # Progress callback
    def show_progress(frame_num, total_frames):
        if frame_num % 30 == 0:  # Update every 30 frames
            progress = (frame_num / total_frames) * 100
            print(
                f"   Progress: {frame_num}/{total_frames} ({progress:.1f}%)", end="\r"
            )

    results = video_processor.process_video_simple(
        video_path=video_path,
        output_path=str(output_path),
        confidence=0.5,
        sample_rate=1,  # Process every frame
        progress_callback=show_progress,
    )

    # Display results
    print("\n\n✅ Processing complete!")
    print(f"\n📊 Results:")
    print(f"   - Frames processed: {results['frames_processed']}")
    print(f"   - Total detections: {results['total_detections']}")
    print(f"   - Avg detections/frame: {results['avg_detections_per_frame']:.2f}")
    print(f"   - Processing time: {results['processing_time_ms'] / 1000:.2f}s")
    print(f"   - Avg time/frame: {results['avg_time_per_frame']:.2f}ms")
    print(f"   - Output saved to: {results['output_path']}")

    # Show class distribution
    class_counts = {}
    for det in results["detections"]:
        cls = det["class"]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    if class_counts:
        print(f"\n📦 Objects detected:")
        for cls, count in sorted(
            class_counts.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"   - {cls}: {count}")


def process_video_with_tracking_example(video_path: str, output_dir: str = "./output"):
    """
    Example 2: Video processing with object tracking

    Best for: Moving objects, crowded scenes, tracking specific objects over time
    """
    print("\n" + "=" * 60)
    print("Example 2: Detection with Object Tracking")
    print("=" * 60)

    # Initialize services
    print("🔧 Initializing YOLO service...")
    yolo_service = YOLOService(model_path="./data/models/yolo/yolov8n.pt", device="cpu")

    video_processor = VideoProcessor(yolo_service)

    # Get video info
    video_info = video_processor.get_video_info(video_path)
    print(
        f"📹 Video: {video_info['total_frames']} frames, {video_info['duration']:.2f}s"
    )

    # Process video with tracking
    output_path = Path(output_dir) / f"{Path(video_path).stem}_tracked.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n⏳ Processing with object tracking...")
    print(f"   This maintains unique IDs for each object across frames")

    def show_progress(frame_num, total_frames):
        if frame_num % 30 == 0:
            progress = (frame_num / total_frames) * 100
            print(
                f"   Progress: {frame_num}/{total_frames} ({progress:.1f}%)", end="\r"
            )

    results = video_processor.process_video_with_tracking(
        video_path=video_path,
        output_path=str(output_path),
        confidence=0.5,
        progress_callback=show_progress,
    )

    # Display results
    print("\n\n✅ Tracking complete!")
    print(f"\n📊 Results:")
    print(f"   - Total frames: {results['total_frames']}")
    print(f"   - Total detections: {results['total_detections']}")
    print(f"   - Unique tracked objects: {results['unique_tracks']}")
    print(f"   - Avg detections/frame: {results['avg_detections_per_frame']:.2f}")
    print(f"   - Processing time: {results['processing_time_ms'] / 1000:.2f}s")
    print(f"   - Output saved to: {results['output_path']}")

    # Show track information
    if results["tracks"]:
        print(f"\n🎯 Tracked Objects:")
        for track_id, track_detections in list(results["tracks"].items())[
            :10
        ]:  # Show first 10
            duration = results["track_durations"][track_id]
            obj_class = track_detections[0]["class"]
            avg_conf = sum(d["confidence"] for d in track_detections) / len(
                track_detections
            )
            print(
                f"   - Track #{track_id}: {obj_class} (appeared in {duration} frames, avg conf: {avg_conf:.2f})"
            )

        if len(results["tracks"]) > 10:
            print(f"   ... and {len(results['tracks']) - 10} more tracks")


def extract_key_frames_example(video_path: str, output_dir: str = "./output"):
    """
    Example 3: Extract and analyze key frames

    Best for: Quick preview, thumbnail generation, summary creation
    """
    print("\n" + "=" * 60)
    print("Example 3: Key Frame Extraction")
    print("=" * 60)

    # Initialize services
    print("🔧 Initializing services...")
    yolo_service = YOLOService(model_path="./data/models/yolo/yolov8n.pt", device="cpu")

    video_processor = VideoProcessor(yolo_service)

    # Extract key frames
    print(f"📹 Extracting 10 key frames from video...")
    key_frames = video_processor.extract_key_frames(video_path, num_frames=10)

    print(f"✅ Extracted {len(key_frames)} frames")

    # Process each key frame
    output_dir_path = Path(output_dir) / "keyframes"
    output_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"\n🔍 Detecting objects in key frames...")

    all_detections = []
    for i, (frame_num, frame) in enumerate(key_frames):
        detections = yolo_service.detect(frame, confidence=0.5)

        # Draw detections
        annotated_frame = yolo_service.draw_detections(frame, detections)

        # Save frame
        import cv2

        output_path = output_dir_path / f"frame_{frame_num:05d}.jpg"
        cv2.imwrite(str(output_path), annotated_frame)

        print(
            f"   Frame {frame_num}: {len(detections)} objects detected -> {output_path}"
        )
        all_detections.extend(detections)

    # Summary
    class_counts = {}
    for det in all_detections:
        cls = det.class_name
        class_counts[cls] = class_counts.get(cls, 0) + 1

    print(f"\n📊 Summary across {len(key_frames)} frames:")
    print(f"   - Total detections: {len(all_detections)}")
    print(f"   - Objects found: {', '.join(class_counts.keys())}")


def quick_preview_example(video_path: str, output_dir: str = "./output"):
    """
    Example 4: Quick preview (process every Nth frame)

    Best for: Long videos, quick analysis, testing
    """
    print("\n" + "=" * 60)
    print("Example 4: Quick Preview (Every 30th frame)")
    print("=" * 60)

    # Initialize services
    print("🔧 Initializing services...")
    yolo_service = YOLOService(model_path="./data/models/yolo/yolov8n.pt", device="cpu")

    video_processor = VideoProcessor(yolo_service)

    # Get video info
    video_info = video_processor.get_video_info(video_path)
    print(f"📹 Video: {video_info['total_frames']} frames")
    print(f"   Will process ~{video_info['total_frames'] // 30} frames (every 30th)")

    # Create quick preview
    output_path = Path(output_dir) / f"{Path(video_path).stem}_preview.mp4"

    print(f"\n⚡ Creating quick preview (much faster)...")

    results = video_processor.quick_preview(
        video_path=video_path,
        output_path=str(output_path),
        sample_every=30,  # Process 1 frame per second at 30fps
        confidence=0.5,
    )

    print(f"\n✅ Preview created!")
    print(f"   - Frames processed: {results['frames_processed']}")
    print(f"   - Processing time: {results['processing_time_ms'] / 1000:.2f}s")
    print(f"   - Output: {results['output_path']}")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO Video Detection Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all examples
    python examples/video_detection_example.py --video test.mp4

    # Run specific example
    python examples/video_detection_example.py --video test.mp4 --example simple

    # Specify output directory
    python examples/video_detection_example.py --video test.mp4 --output ./results
        """,
    )

    parser.add_argument("--video", required=True, help="Path to input video file")

    parser.add_argument(
        "--example",
        choices=["simple", "tracking", "keyframes", "preview", "all"],
        default="all",
        help="Which example to run (default: all)",
    )

    parser.add_argument(
        "--output",
        default="./output",
        help="Output directory for results (default: ./output)",
    )

    args = parser.parse_args()

    # Check if video exists
    if not Path(args.video).exists():
        print(f"❌ Error: Video file not found: {args.video}")
        sys.exit(1)

    print("🎬 YOLO Video Detection Examples")
    print(f"📹 Input video: {args.video}")
    print(f"📁 Output directory: {args.output}")

    try:
        # Run selected examples
        if args.example == "all" or args.example == "simple":
            process_video_simple_example(args.video, args.output)

        if args.example == "all" or args.example == "tracking":
            process_video_with_tracking_example(args.video, args.output)

        if args.example == "all" or args.example == "keyframes":
            extract_key_frames_example(args.video, args.output)

        if args.example == "all" or args.example == "preview":
            quick_preview_example(args.video, args.output)

        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print(f"📁 Check the output directory: {args.output}")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
