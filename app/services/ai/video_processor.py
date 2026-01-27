"""
Video Processing Service for YOLO Object Detection

This service handles video processing including frame extraction,
object detection, tracking, and annotated video generation.
"""

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Service for processing videos with YOLO object detection
    Handles frame extraction, detection, tracking, and output generation
    """

    def __init__(self, yolo_service):
        """
        Initialize video processor

        Args:
            yolo_service: Instance of YOLOService for object detection
        """
        self.yolo_service = yolo_service
        self.trackers = {}  # Store trackers for different tracking methods

    def get_video_info(self, video_path: str) -> Dict:
        """
        Extract video metadata

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with video information

        Raises:
            ValueError: If video cannot be opened
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            / cap.get(cv2.CAP_PROP_FPS),
            "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
        }

        cap.release()
        return info

    def validate_video(self, video_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that video file is readable and processable

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (is_valid, error_message)
        """
        path = Path(video_path)

        if not path.exists():
            return False, f"Video file does not exist: {video_path}"

        if not path.is_file():
            return False, f"Path is not a file: {video_path}"

        # Try to open and read one frame
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            cap.release()
            return False, "Could not open video file"

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return False, "Could not read frames from video"

        return True, None

    def extract_frames(
        self, video_path: str, sample_rate: int = 1
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Extract frames from video

        Args:
            video_path: Path to video file
            sample_rate: Process every Nth frame (1 = all frames)

        Yields:
            Tuple of (frame_number, frame_image)

        Raises:
            ValueError: If video cannot be opened
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        frame_number = 0

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                if frame_number % sample_rate == 0:
                    yield frame_number, frame

                frame_number += 1
        finally:
            cap.release()

    def process_video_simple(
        self,
        video_path: str,
        output_path: str,
        confidence: float = 0.5,
        sample_rate: int = 1,
        progress_callback: Optional[callable] = None,
    ) -> Dict:
        """
        Process video with simple frame-by-frame detection (no tracking)

        Args:
            video_path: Path to input video
            output_path: Path to save annotated video
            confidence: Detection confidence threshold
            sample_rate: Process every Nth frame
            progress_callback: Optional callback function(frame_num, total_frames)

        Returns:
            Dictionary with processing results and statistics

        Raises:
            ValueError: If video cannot be processed
        """
        # Validate video first
        is_valid, error_msg = self.validate_video(video_path)
        if not is_valid:
            raise ValueError(error_msg)

        # Get video info
        video_info = self.get_video_info(video_path)

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Setup video writer with fallback codecs
        fourcc_options = [
            cv2.VideoWriter_fourcc(*"mp4v"),  # MPEG-4
            cv2.VideoWriter_fourcc(*"X264"),  # H.264
            cv2.VideoWriter_fourcc(*"avc1"),  # H.264 alternative
        ]

        out = None
        for fourcc in fourcc_options:
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                video_info["fps"],
                (video_info["width"], video_info["height"]),
            )
            if out.isOpened():
                logger.info(f"Using codec: {fourcc}")
                break

        if out is None or not out.isOpened():
            raise ValueError("Could not initialize video writer with any codec")

        all_detections = []
        start_time = time.time()
        frames_processed = 0

        try:
            for frame_num, frame in self.extract_frames(video_path, sample_rate):
                # Detect objects in frame
                detections = self.yolo_service.detect(frame, confidence=confidence)

                # Draw detections on frame
                annotated_frame = self.yolo_service.draw_detections(frame, detections)

                # Write annotated frame
                out.write(annotated_frame)

                # Store detections with frame number
                for detection in detections:
                    detection_with_frame = detection.to_dict()
                    detection_with_frame["frame"] = frame_num
                    all_detections.append(detection_with_frame)

                frames_processed += 1

                # Progress callback
                if progress_callback:
                    progress_callback(frame_num, video_info["total_frames"])

        finally:
            out.release()

        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        return {
            "total_frames": video_info["total_frames"],
            "frames_processed": frames_processed,
            "total_detections": len(all_detections),
            "avg_detections_per_frame": len(all_detections) / frames_processed
            if frames_processed > 0
            else 0,
            "processing_time_ms": processing_time,
            "avg_time_per_frame": processing_time / frames_processed
            if frames_processed > 0
            else 0,
            "detections": all_detections,
            "output_path": output_path,
        }

    def process_video_with_tracking(
        self,
        video_path: str,
        output_path: str,
        confidence: float = 0.5,
        progress_callback: Optional[callable] = None,
    ) -> Dict:
        """
        Process video with object tracking using YOLO's built-in tracker

        Args:
            video_path: Path to input video
            output_path: Path to save annotated video
            confidence: Detection confidence threshold
            progress_callback: Optional callback function(frame_num, total_frames)

        Returns:
            Dictionary with tracking results and statistics

        Raises:
            ValueError: If video cannot be processed
        """
        # Validate video first
        is_valid, error_msg = self.validate_video(video_path)
        if not is_valid:
            raise ValueError(error_msg)

        # Get video info
        video_info = self.get_video_info(video_path)

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            video_info["fps"],
            (video_info["width"], video_info["height"]),
        )

        if not out.isOpened():
            raise ValueError("Could not initialize video writer")

        # Use YOLO's built-in tracking
        cap = cv2.VideoCapture(video_path)

        all_tracks = defaultdict(list)  # track_id -> list of detections
        frame_detections = []
        start_time = time.time()
        frame_num = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Track objects (YOLO will maintain track IDs across frames)
                results = self.yolo_service.model.track(
                    frame,
                    conf=confidence,
                    persist=True,  # Persist tracks between frames
                    verbose=False,
                )

                # Process results
                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes

                    for i, box in enumerate(boxes):
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])

                        # Get track ID if available
                        track_id = int(box.id[0]) if box.id is not None else None

                        detection = {
                            "frame": frame_num,
                            "track_id": track_id,
                            "class": self.yolo_service.model.names[cls_id],
                            "class_id": cls_id,
                            "confidence": conf,
                            "bbox": {
                                "x1": float(x1),
                                "y1": float(y1),
                                "x2": float(x2),
                                "y2": float(y2),
                                "width": float(x2 - x1),
                                "height": float(y2 - y1),
                            },
                        }

                        frame_detections.append(detection)

                        if track_id is not None:
                            all_tracks[track_id].append(detection)

                        # Draw on frame
                        cv2.rectangle(
                            frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0, 255, 0),
                            2,
                        )

                        # Draw label with track ID
                        label = f"ID:{track_id} {detection['class']} {conf:.2f}"
                        cv2.putText(
                            frame,
                            label,
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

                # Write frame
                out.write(frame)

                frame_num += 1

                if progress_callback:
                    progress_callback(frame_num, video_info["total_frames"])

        finally:
            cap.release()
            out.release()

        processing_time = (time.time() - start_time) * 1000

        # Calculate tracking statistics
        track_durations = {
            track_id: len(detections) for track_id, detections in all_tracks.items()
        }

        return {
            "total_frames": frame_num,
            "total_detections": len(frame_detections),
            "unique_tracks": len(all_tracks),
            "avg_detections_per_frame": len(frame_detections) / frame_num
            if frame_num > 0
            else 0,
            "processing_time_ms": processing_time,
            "avg_time_per_frame": processing_time / frame_num if frame_num > 0 else 0,
            "tracks": dict(all_tracks),
            "track_durations": track_durations,
            "detections": frame_detections,
            "output_path": output_path,
        }

    def extract_key_frames(
        self, video_path: str, num_frames: int = 10
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Extract evenly-spaced key frames from video

        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract

        Returns:
            List of (frame_number, frame_image) tuples

        Raises:
            ValueError: If video cannot be opened
        """
        video_info = self.get_video_info(video_path)
        total_frames = video_info["total_frames"]

        # Calculate frame indices to extract
        if num_frames >= total_frames:
            frame_indices = list(range(total_frames))
        else:
            step = total_frames // num_frames
            frame_indices = [i * step for i in range(num_frames)]

        cap = cv2.VideoCapture(video_path)
        key_frames = []

        try:
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    key_frames.append((frame_idx, frame))
        finally:
            cap.release()

        return key_frames

    def create_summary_video(
        self,
        video_path: str,
        output_path: str,
        detections: List[Dict],
        confidence: float = 0.5,
    ) -> str:
        """
        Create a shorter summary video showing only frames with detections

        Args:
            video_path: Path to input video
            output_path: Path to save summary video
            detections: List of detections with frame numbers
            confidence: Minimum confidence to include frame

        Returns:
            Path to summary video

        Raises:
            ValueError: If no frames with detections or video cannot be processed
        """
        # Find frames with significant detections
        frames_with_detections = set()
        for det in detections:
            if det.get("confidence", 0) >= confidence:
                frames_with_detections.add(det["frame"])

        if not frames_with_detections:
            raise ValueError("No frames with detections above confidence threshold")

        frames_with_detections = sorted(frames_with_detections)

        # Get video info
        video_info = self.get_video_info(video_path)

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            video_info["fps"],
            (video_info["width"], video_info["height"]),
        )

        if not out.isOpened():
            raise ValueError("Could not initialize video writer")

        cap = cv2.VideoCapture(video_path)

        try:
            for frame_num in frames_with_detections:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()

                if ret:
                    # Get detections for this frame
                    frame_dets = [d for d in detections if d.get("frame") == frame_num]

                    # Draw detections
                    for det in frame_dets:
                        bbox = det.get("bbox", {})
                        x1, y1 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0))
                        x2, y2 = int(bbox.get("x2", 0)), int(bbox.get("y2", 0))

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        label = f"{det.get('class', 'unknown')} {det.get('confidence', 0):.2f}"
                        cv2.putText(
                            frame,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

                    out.write(frame)
        finally:
            cap.release()
            out.release()

        return output_path

    def quick_preview(
        self,
        video_path: str,
        output_path: str,
        sample_every: int = 30,
        confidence: float = 0.5,
    ) -> Dict:
        """
        Create a quick preview by processing fewer frames
        Useful for fast analysis of long videos

        Args:
            video_path: Path to input video
            output_path: Path to save preview video
            sample_every: Process 1 frame every N frames (e.g., 30 = 1 fps at 30fps video)
            confidence: Detection confidence threshold

        Returns:
            Dictionary with processing results
        """
        return self.process_video_simple(
            video_path=video_path,
            output_path=output_path,
            confidence=confidence,
            sample_rate=sample_every,
        )
