import os
import numpy as np
import cv2
import pytest
from video_ql.visualization import VideoVisualizer
from video_ql.models import Label


# Get the absolute path to the test video
def get_test_video_path():
    return os.path.join(os.path.dirname(__file__), "forklift.small.mp4")


def test_visualize_results():
    # Create a test frame and analysis
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    analysis = Label(
        timestamp=2.5,
        results={
            "action": "moving",
            "object_detected": "forklift",
            "safety_status": "normal",
        },
        error=None,
    )

    # Call the visualize_results method
    result = VideoVisualizer.visualize_results(frame, analysis)

    # Check that result is an image with the same dimensions as input
    assert isinstance(result, np.ndarray)
    assert result.shape == frame.shape

    # Test with error
    analysis_with_error = Label(
        timestamp=2.5, results={"action": "moving"}, error="Detection failed"
    )
    result_with_error = VideoVisualizer.visualize_results(
        frame, analysis_with_error
    )
    assert isinstance(result_with_error, np.ndarray)


def test_create_tile_image_from_frames():
    # Create test frames
    frames = [
        {"frame": np.ones((100, 100, 3), dtype=np.uint8) * i * 50}
        for i in range(4)
    ]

    # Test with 2x2 tile shape
    tile_image = VideoVisualizer.create_tile_image_from_frames(frames, (2, 2))
    assert isinstance(tile_image, np.ndarray)
    assert tile_image.shape == (200, 200, 3)

    # Test with empty frames list
    empty_tile = VideoVisualizer.create_tile_image_from_frames([], (2, 2))
    assert empty_tile.shape == (100, 100, 3)


def test_extract_frames():
    video_path = get_test_video_path()

    # Skip test if video file doesn't exist
    if not os.path.exists(video_path):
        pytest.skip(f"Test video not found: {video_path}")

    # Test parameters
    start_idx = 0
    count = 5
    stride = 2
    max_resolution = (320, 240)

    # Extract frames
    frames = VideoVisualizer.extract_frames(
        video_path, start_idx, count, stride, max_resolution
    )

    # Check we got the expected number of frames
    assert len(frames) <= count  # Could be less if video is shorter

    # Check frame structure
    for frame_data in frames:
        assert "frame" in frame_data
        assert "timestamp" in frame_data
        assert isinstance(frame_data["frame"], np.ndarray)
        assert frame_data["frame"].shape[0] <= max_resolution[1]  # height
        assert frame_data["frame"].shape[1] <= max_resolution[0]  # width

    # Test with invalid file
    with pytest.raises(ValueError):
        VideoVisualizer.extract_frames(
            "nonexistent_file.mp4", 0, 1, 1, (640, 480)
        )


def test_end_to_end_visualization():
    """Test the complete visualization workflow with the sample video."""
    video_path = get_test_video_path()

    # Skip test if video file doesn't exist
    if not os.path.exists(video_path):
        pytest.skip(f"Test video not found: {video_path}")

    # Extract a few frames
    frames = VideoVisualizer.extract_frames(
        video_path, start_idx=0, count=4, stride=10, max_resolution=(640, 480)
    )

    # Create sample analysis for each frame
    annotated_frames = []
    for i, frame_data in enumerate(frames):
        analysis = Label(
            timestamp=frame_data["timestamp"],
            results={
                "frame_number": i,
                "action": "driving" if i % 2 == 0 else "stopped",
                "safety_status": "normal",
            },
            error=None,
        )

        # Apply visualization
        vis_frame = VideoVisualizer.visualize_results(
            frame_data["frame"], analysis
        )
        annotated_frames.append(
            {"frame": vis_frame, "timestamp": frame_data["timestamp"]}
        )

    # Create a tile image
    tile_image = VideoVisualizer.create_tile_image_from_frames(
        annotated_frames, (2, 2)
    )

    # Verify the tile was created with expected dimensions
    assert isinstance(tile_image, np.ndarray)
    assert len(tile_image.shape) == 3  # Should be a color image

    # Optional: save the visualization for manual inspection
    # output_path = os.path.join(os.path.dirname(__file__), "test_visualization.jpg")
    # cv2.imwrite(output_path, tile_image)
