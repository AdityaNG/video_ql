"""
video_ql utils module.
"""

import base64
import hashlib
import io
import os
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image


def video_hash(video_path: str) -> str:
    """
    Generate a quick fingerprint for the video file using a constant-time
    approach. This is not a true hash but a sufficient fingerprint for
    most video identification purposes.
    """
    file_hash = hashlib.md5()
    file_size = os.path.getsize(video_path)

    # If file is too small, just hash the whole thing
    if file_size <= 16384:  # 16KB
        with open(video_path, "rb") as f:
            file_hash.update(f.read())
        return file_hash.hexdigest()

    chunk_size = 4096  # 4KB chunks

    with open(video_path, "rb") as f:
        # First chunk
        file_hash.update(f.read(chunk_size))

        # Middle chunks (3 samples)
        for position in [0.25, 0.5, 0.75]:
            f.seek(int(file_size * position))
            file_hash.update(f.read(chunk_size))

        # Last chunk
        f.seek(max(0, file_size - chunk_size))
        file_hash.update(f.read(chunk_size))

        # Add file size to the hash to differentiate similarly structured
        # but different-sized videos
        file_hash.update(str(file_size).encode())

    return file_hash.hexdigest()


def get_length_of_video(video_path: str) -> int:
    """Get the number of frames in a video using FFmpeg for estimation
    and then refining"""
    estimated_frames = None

    # Use FFmpeg to get an estimate of the frame count
    try:
        if shutil.which("ffprobe") is not None:
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_packets",
                "-show_entries",
                "stream=nb_read_packets",
                "-of",
                "csv=p=0",
                video_path,
            ]
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            estimated_frames = int(result.stdout.strip())
    except (ValueError, subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"FFprobe method failed: {e}. Falling back to OpenCV.")

    if estimated_frames is None:
        # Fallback if ffprobe fails
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        estimated_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    # Now refine the estimate by seeking near the end and finding the
    # actual last frame
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Try seeking to the estimated end
    seek_frame = max(0, estimated_frames - 5)

    # Binary search approach to find the actual end
    frame_min = 0
    frame_max = estimated_frames * 2  # Allow for estimates that are too low
    actual_frame_count = 0

    while frame_min <= frame_max:
        seek_frame = (frame_min + frame_max) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, seek_frame)
        ret, _ = cap.read()

        if ret:
            # This frame exists, so the actual count is at least seek_frame + 1
            actual_frame_count = seek_frame + 1
            frame_min = seek_frame + 1
        else:
            # This frame doesn't exist, so the end is before this
            frame_max = seek_frame - 1

    cap.release()
    return actual_frame_count


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
