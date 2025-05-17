import os
import pytest
import cv2
import numpy as np
from unittest.mock import MagicMock, patch

from video_ql.base import VideoQL
from video_ql.models import Query, Label, QueryConfig, QueryCondition


# Get the absolute path to the test video
def get_test_video_path():
    return os.path.join(os.path.dirname(__file__), "forklift.small.mp4")


# Check if OpenAI API key is available
has_openai_key = os.environ.get("OPENAI_API_KEY", "").startswith("sk-")

# Basic test queries
sample_queries = [
    Query(
        query="Is the driver present in the forklift?", options=["yes", "no"]
    ),
    Query(
        query="Where is the forklift currently at?",
        options=["Truck", "Warehouse", "Charging"],
    ),
]


def test_videoql_init():
    """Test initialization of VideoQL class"""
    video_path = get_test_video_path()
    video_ql = VideoQL(
        video_path=video_path,
        queries=sample_queries,
        context="Test context",
        disable_cache=True,  # Disable cache for testing
    )

    assert video_ql.video_path == video_path
    assert video_ql.queries == sample_queries
    assert video_ql.context == "Test context"
    assert video_ql.disable_cache == True
    assert video_ql.model_name == "gpt-4o-mini"
    assert hasattr(video_ql, "visualizer")
    assert hasattr(video_ql, "frame_model")
    assert hasattr(video_ql, "parser")


def test_extract_frames():
    """Test extracting frames from video"""
    video_path = get_test_video_path()
    video_ql = VideoQL(
        video_path=video_path, queries=sample_queries, disable_cache=True
    )

    frames = video_ql.extract_frames(start_idx=0, count=3)

    assert len(frames) == 3
    for frame in frames:
        assert "frame" in frame
        assert "timestamp" in frame
        assert isinstance(frame["frame"], np.ndarray)
        assert isinstance(frame["timestamp"], float)


def test_create_tile_image():
    """Test creating a tile image from frames"""
    video_path = get_test_video_path()
    video_ql = VideoQL(
        video_path=video_path, queries=sample_queries, disable_cache=True
    )

    tile_image = video_ql.create_tile_image(start_idx=0)

    assert isinstance(tile_image, np.ndarray)
    # Check dimensions - should match the max_resolution approximately
    height, width, _ = tile_image.shape
    assert (
        height
        <= video_ql.config.max_resolution[1]
        * video_ql.config.tile_frames[1]
        * 1.1
    )
    assert (
        width
        <= video_ql.config.max_resolution[0]
        * video_ql.config.tile_frames[0]
        * 1.1
    )


def test_create_tile_image_from_frames():
    """Test creating a tile image from specific frames"""
    video_path = get_test_video_path()
    video_ql = VideoQL(
        video_path=video_path, queries=sample_queries, disable_cache=True
    )

    frames = video_ql.extract_frames(start_idx=0, count=9)
    tile_image = video_ql.create_tile_image_from_frames(frames)

    assert isinstance(tile_image, np.ndarray)
    # Check dimensions
    height, width, _ = tile_image.shape
    assert (
        height
        <= video_ql.config.max_resolution[1]
        * video_ql.config.tile_frames[1]
        * 1.1
    )
    assert (
        width
        <= video_ql.config.max_resolution[0]
        * video_ql.config.tile_frames[0]
        * 1.1
    )


def test_len():
    """Test __len__ method returns expected number of processable frames"""
    video_path = get_test_video_path()
    video_ql = VideoQL(
        video_path=video_path, queries=sample_queries, disable_cache=True
    )

    # Get actual video length
    cap = cv2.VideoCapture(video_path)
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # VideoQL len should be less than or equal to video frames
    assert len(video_ql) <= video_frames


@pytest.mark.skipif(not has_openai_key, reason="OpenAI API key not available")
def test_analyze_frame():
    """Test analyzing a frame with the vision model"""
    video_path = get_test_video_path()
    video_ql = VideoQL(
        video_path=video_path, queries=sample_queries, disable_cache=True
    )

    frames = video_ql.extract_frames(start_idx=0, count=1)
    frame = {"frame": frames[0]["frame"], "timestamp": frames[0]["timestamp"]}

    # Analyze the frame
    result = video_ql._analyze_frame(frame)

    assert isinstance(result, Label)
    assert result.timestamp == frame["timestamp"]
    assert isinstance(result.results, dict)


@patch("video_ql.base.ChatOpenAI")
def test_analyze_frame_mock(mock_chat):
    """Test _analyze_frame with a mocked LLM to avoid API calls"""
    # Configure the mock
    mock_instance = MagicMock()
    mock_chat.return_value = mock_instance
    mock_response = MagicMock()
    mock_response.content = '{"is_the_driver_present_in_the_forklift": "yes", "where_is_the_forklift_currently_at": "Warehouse", "timestamp": 0.5}'
    mock_instance.invoke.return_value = mock_response

    video_path = get_test_video_path()
    video_ql = VideoQL(
        video_path=video_path, queries=sample_queries, disable_cache=True
    )

    frames = video_ql.extract_frames(start_idx=0, count=1)
    frame = {"frame": frames[0]["frame"], "timestamp": 0.5}

    # Analyze the frame
    result = video_ql._analyze_frame(frame)

    assert isinstance(result, Label)
    assert result.timestamp == 0.5
    assert "is_the_driver_present_in_the_forklift" in result.results
    assert result.results["is_the_driver_present_in_the_forklift"] == "yes"


def test_cache_operations():
    """Test cache loading and saving operations"""
    video_path = get_test_video_path()

    # Create a temporary cache dir
    import tempfile

    temp_dir = tempfile.mkdtemp()

    # Create VideoQL with the temp cache dir
    video_ql = VideoQL(
        video_path=video_path,
        queries=sample_queries,
        cache_dir=temp_dir,
        disable_cache=False,
    )

    # Mock some cache data
    mock_label = Label(timestamp=0.5, results={"test_key": "test_value"})
    video_ql._VideoQL__cache[0] = mock_label

    # Save cache
    video_ql._update_cache(0, mock_label)

    # Create a new instance that should load the cache
    video_ql2 = VideoQL(
        video_path=video_path,
        queries=sample_queries,
        cache_dir=temp_dir,
        disable_cache=False,
    )

    # Check if cache was loaded
    assert 0 in video_ql2._VideoQL__cache
    assert video_ql2._VideoQL__cache[0].timestamp == 0.5
    assert video_ql2._VideoQL__cache[0].results["test_key"] == "test_value"

    # Clean up
    import shutil

    shutil.rmtree(temp_dir)


# @patch("video_ql.base.VideoQL._analyze_frame")
# def test_getitem(mock_analyze_frame):
#     """Test __getitem__ with a mocked _analyze_frame to avoid API calls"""
#     # Configure the mock
#     mock_label = Label(timestamp=0.5, results={"test_key": "test_value"})
#     mock_analyze_frame.return_value = mock_label

#     video_path = get_test_video_path()
#     video_ql = VideoQL(
#         video_path=video_path, queries=sample_queries, disable_cache=True
#     )

#     assert len(video_ql) > 0

#     # Get an item
#     result = video_ql[0]

#     assert isinstance(result, Label)
#     assert result.timestamp == 0.5
#     assert result.results["test_key"] == "test_value"


# @pytest.mark.skipif(not has_openai_key, reason="OpenAI API key not available")
# def test_get_frame_analysis():
#     """Test getting frame analysis at a specific timestamp"""
#     video_path = get_test_video_path()
#     video_ql = VideoQL(
#         video_path=video_path, queries=sample_queries, disable_cache=True
#     )

#     # Get analysis at timestamp 1.0
#     with patch.object(
#         video_ql,
#         "__getitem__",
#         return_value=Label(timestamp=1.0, results={"test": "value"}),
#     ):
#         result = video_ql.get_frame_analysis(1.0)

#         assert isinstance(result, Label)
#         assert result.timestamp == 1.0
#         assert "test" in result.results


@patch("video_ql.base.VideoQL.__getitem__")
@patch("video_ql.base.matches_query")
def test_query_video(mock_matches_query, mock_getitem):
    """Test querying video based on a query configuration"""
    # Configure mocks
    mock_getitem.return_value = Label(timestamp=0.5, results={"test": "value"})
    mock_matches_query.return_value = True

    video_path = get_test_video_path()
    video_ql = VideoQL(
        video_path=video_path, queries=sample_queries, disable_cache=True
    )

    # Set up a minimal cache for testing
    video_ql._VideoQL__cache = {
        0: Label(timestamp=0.5, results={"test": "value"})
    }

    # Create query config
    query_config = QueryConfig(
        queries=[
            QueryCondition(
                query="Is the driver present in the forklift?", options=["yes"]
            )
        ]
    )

    # Query the video
    matching_frames = video_ql.query_video(query_config)

    assert isinstance(matching_frames, list)
    assert (
        len(matching_frames) > 0
    )  # Should match since we mocked matches_query to True
