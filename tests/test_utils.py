"""
Tests for video_ql utils module.
"""

import io
import os
import base64
import tempfile
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


def test_get_length_of_video(sample_video_file):
    """Test the get_length_of_video function."""
    # Our sample video has 10 frames
    assert get_length_of_video(sample_video_file) == 10

    # Test with non-existent file
    with pytest.raises(ValueError):
        get_length_of_video("non_existent_file.mp4")


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
