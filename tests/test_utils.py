"""
Tests for video_ql utils module.
"""

import io
import os
import base64
import tempfile
import time
import cv2
import numpy as np
import pytest
from PIL import Image

from video_ql.utils import (
    video_hash,
    get_length_of_video,
    get_video_fps,
    encode_image,
)


@pytest.fixture
def sample_video_file():
    """Creates a temporary test video file."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp:
        temp_filename = temp.name

    # Create a simple test video
    fps = 25
    width, height = 320, 240
    out = cv2.VideoWriter(
        temp_filename, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    # Add 10 frames to the video
    for i in range(10):
        # Create a simple colored frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * (i * 25)
        out.write(frame)

    out.release()

    yield temp_filename

    # Clean up after test
    if os.path.exists(temp_filename):
        os.remove(temp_filename)


def test_video_hash(sample_video_file):
    """Test the video_hash function."""
    # Get hash of the video
    hash_result = video_hash(sample_video_file)

    # Hash should be a string with 32 hex characters (MD5)
    assert isinstance(hash_result, str)
    assert len(hash_result) == 32

    # Same file should produce same hash
    assert video_hash(sample_video_file) == hash_result

    # Different content should produce different hash
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp:
        temp.write(b"different content")
        different_file = temp.name

    try:
        assert video_hash(different_file) != hash_result
    finally:
        if os.path.exists(different_file):
            os.remove(different_file)


def test_video_hash_performance():
    """Test that video_hash processes large files quickly."""
    # Create a 40MB temporary file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp:
        # Write 40MB of data (in chunks to avoid memory issues)
        chunk_size = 1024 * 1024  # 1MB
        for _ in range(40):
            temp.write(os.urandom(chunk_size))

        temp_filename = temp.name

    try:
        # Time the hash operation
        start_time = time.time()
        hash_result = video_hash(temp_filename)
        end_time = time.time()

        # Calculate duration in milliseconds
        duration_ms = (end_time - start_time) * 1000

        # Verify the hash is valid
        assert isinstance(hash_result, str)
        assert len(hash_result) == 32

        # Assert that the operation runs in the stipulated time
        assert (
            duration_ms < 10
        ), f"Video hashing took {duration_ms:.2f}ms, which exceeds the time limit"

    finally:
        # Clean up
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


def test_get_length_of_video(sample_video_file):
    """Test the get_length_of_video function."""
    # Our sample video has 10 frames
    assert get_length_of_video(sample_video_file) == 10

    # Test with non-existent file
    with pytest.raises(ValueError):
        get_length_of_video("non_existent_file.mp4")


def test_get_length_of_video_performance():
    """Test that get_length_of_video processes large HD videos quickly."""
    # Use a cached video file in the user's cache directory
    cache_dir = os.path.expanduser("~/.cache")
    os.makedirs(cache_dir, exist_ok=True)
    cached_video_path = os.path.join(cache_dir, "video_ql_test_video.mp4")

    # Video specifications
    fps = 60
    width, height = 1920, 1080
    num_test_frames = fps * 60 * 1  # 1 min

    # Only create the test video if it doesn't exist
    if not os.path.exists(cached_video_path):
        # Create a short HD test video (1920x1080) with random noise
        out = cv2.VideoWriter(
            cached_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        # Add frames with random noise
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        for _ in range(num_test_frames):
            out.write(frame)

        out.release()

    try:
        # Time the operation
        start_time = time.time()
        frame_count = get_length_of_video(cached_video_path)
        end_time = time.time()

        # Calculate time per frame in milliseconds
        duration_ms = (end_time - start_time) * 1000

        # Verify correct frame count
        assert frame_count == num_test_frames

        # Assert that the extrapolated time is under 100ms
        assert (
            duration_ms < 3000
        ), f"Estimated time for {num_test_frames} frames: {duration_ms:.2f}ms exceeds time limit"

    except Exception as e:
        # Don't delete the cached video on error to help with debugging
        raise e


def test_get_video_fps(sample_video_file):
    """Test the get_video_fps function."""
    # Our sample video has 25 fps
    assert get_video_fps(sample_video_file) == 25.0

    # Test with non-existent file
    with pytest.raises(ValueError):
        get_video_fps("non_existent_file.mp4")


def test_encode_image():
    """Test the encode_image function."""
    # Create a simple test image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:50, :50] = [255, 0, 0]  # Red square

    # Encode the image
    encoded = encode_image(img)

    # Result should be a base64 string
    assert isinstance(encoded, str)

    # Verify it's a valid base64 string
    try:
        decoded = base64.b64decode(encoded)
        # Should be able to open with PIL
        Image.open(io.BytesIO(decoded))
        valid_base64 = True
    except Exception:
        valid_base64 = False

    assert valid_base64, "Encoded image is not a valid base64 JPEG image"
