"""Visualization module for video_ql."""

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .models import Label


class VideoVisualizer:
    """Class responsible for video visualization in video_ql."""

    @staticmethod
    def visualize_results(frame: np.ndarray, analysis: Label) -> np.ndarray:
        """
        Overlay analysis results on the frame with a clean, professional look.
        """
        # Create a copy of the frame
        vis_frame = frame.copy()
        h, w = vis_frame.shape[:2]

        # Format results for display
        status_info = {}
        for key, value in analysis.results.items():
            if key != "timestamp":
                # Convert snake_case to Title Case
                formatted_key = key.replace("_", " ").title()
                status_info[formatted_key] = value

        # Create a blue semi-transparent box in the top-left corner
        box_height = 30 * (
            len(status_info) + 1
        )  # Height based on number of items
        box_width = int(0.9 * w)  # Fixed width

        # Create blue box with transparency
        overlay = vis_frame.copy()
        cv2.rectangle(
            overlay,
            (10, 10),
            (10 + box_width, 10 + box_height),
            (255, 0, 0),
            -1,
        )
        vis_frame = cv2.addWeighted(overlay, 0.8, vis_frame, 0.2, 0)

        # Add text to the box in white
        y_pos = 40
        for key, value in status_info.items():
            text = f"{key}: {value}"
            cv2.putText(
                vis_frame,
                text,
                (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            y_pos += 30

        # Add timestamp at the bottom
        cv2.putText(
            vis_frame,
            f"Time: {analysis.timestamp:.2f}s",
            (10, vis_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        # Add error message if present
        if analysis.error:
            cv2.putText(
                vis_frame,
                f"Error: {analysis.error}",
                (10, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                1,
            )

        return vis_frame

    @staticmethod
    def create_tile_image_from_frames(
        frames: list, tile_shape: tuple
    ) -> np.ndarray:
        """Create a tiled image from multiple frames."""
        if not frames:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        # Get dimensions from the first frame
        frame_height, frame_width = frames[0]["frame"].shape[:2]
        rows, cols = tile_shape

        # Calculate the dimensions of the tiled image
        tile_height = rows * frame_height
        tile_width = cols * frame_width

        # Create an empty canvas
        tile_image = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)

        # Fill the canvas with frames
        for i, frame_data in enumerate(frames[: rows * cols]):
            row = i // cols
            col = i % cols

            y_start = row * frame_height
            y_end = y_start + frame_height
            x_start = col * frame_width
            x_end = x_start + frame_width

            tile_image[y_start:y_end, x_start:x_end] = frame_data["frame"]

        return tile_image

    @staticmethod
    def extract_frames(
        video_path: str,
        start_idx: int,
        count: int,
        stride: int,
        max_resolution: Tuple[int, int],
    ) -> List[Dict[str, Any]]:
        """Extract frames from video starting at index"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []  # type: ignore

        # Set position to start_idx
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

        for _ in range(count):
            ret, frame = cap.read()

            if not ret:
                break

            # Resize frame if needed
            if frame is not None:
                h, w = frame.shape[:2]  # type: ignore
                max_w, max_h = max_resolution

                scale = 1.0
                if w > max_w or h > max_h:
                    scale_w = max_w / w
                    scale_h = max_h / h
                    scale = min(scale_w, scale_h)

                if scale < 1.0:
                    new_w, new_h = int(w * scale), int(h * scale)
                    frame = cv2.resize(
                        frame, (new_w, new_h), interpolation=cv2.INTER_AREA
                    )

            timestamp = (start_idx + len(frames)) / video_fps
            frames.append({"frame": frame, "timestamp": timestamp})

            # Stride jump
            for _ in range(stride - 1):
                ret, frame = cap.read()

        cap.release()
        return frames
