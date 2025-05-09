"""
video_ql utils module.
"""

import base64
import hashlib
import io

import cv2
import numpy as np
from PIL import Image


def video_hash(video_path: str) -> str:
    """Generate a hash for the video file"""
    file_hash = hashlib.md5()
    with open(video_path, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(4096), b""):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def get_length_of_video(video_path: str) -> int:
    """Get the number of frames in a video by iterating through all frames"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_count = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        frame_count += 1

    cap.release()
    return frame_count


def get_video_fps(video_path: str) -> float:
    """Get the frames per second (FPS) of a video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()
    return fps


def encode_image(image_array: np.ndarray) -> str:
    """Encode image array to base64 string."""
    image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode("utf-8")
