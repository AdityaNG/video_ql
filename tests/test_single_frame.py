import os
import pytest
import cv2
import numpy as np
from unittest.mock import MagicMock, patch

from video_ql.single_frame import SingleFrameAnalyzer
from video_ql.models import Query, Label


def get_test_image_path():
    """Get the path to a test image."""
    # Use a video frame from the test video as a test image
    video_path = os.path.join(os.path.dirname(__file__), "forklift.small.mp4")
    # Extract first frame and save it
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    image_path = os.path.join(os.path.dirname(__file__), "test_frame.jpg")
    cv2.imwrite(image_path, frame)
    return image_path


# Sample queries for testing
sample_queries = [
    Query(
        query="Is the driver present in the forklift?", options=["yes", "no"]
    ),
    Query(
        query="Where is the forklift currently at?",
        options=["Truck", "Warehouse", "Charging"],
    ),
]


@patch("video_ql.single_frame.VideoQL")
def test_single_frame_analyzer_init(mock_videoql):
    """Test initialization of SingleFrameAnalyzer."""
    # Configure the mock
    mock_instance = MagicMock()
    mock_videoql.return_value = mock_instance

    image_path = get_test_image_path()

    # Create analyzer
    analyzer = SingleFrameAnalyzer(
        image_path=image_path,
        queries=sample_queries,
        context="Test context",
        model_name="test-model",
    )

    # Check initialization
    assert analyzer.image_path == image_path
    assert analyzer.queries == sample_queries
    assert analyzer.context == "Test context"
    assert analyzer.model_name == "test-model"

    # Check if VideoQL was properly initialized
    mock_videoql.assert_called_once()
    args, kwargs = mock_videoql.call_args
    assert kwargs["video_path"] == image_path
    assert kwargs["queries"] == sample_queries
    assert kwargs["context"] == "Test context"
    assert kwargs["model_name"] == "test-model"
    assert kwargs["disable_cache"] is True


@patch("video_ql.single_frame.cv2.imread")
def test_analyze_image_with_invalid_image(mock_imread):
    """Test analyzing with an invalid image path."""
    # Configure mock to return None (simulating failed image load)
    mock_imread.return_value = None

    # Test that ValueError is raised for invalid image
    with pytest.raises(FileNotFoundError):
        analyzer = SingleFrameAnalyzer(
            image_path="invalid_path.jpg", queries=sample_queries
        )


def teardown_module():
    """Clean up any test files."""
    test_image = os.path.join(os.path.dirname(__file__), "test_frame.jpg")
    if os.path.exists(test_image):
        os.remove(test_image)
