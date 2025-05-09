# VideoQL

[![codecov](https://codecov.io/gh/AdityaNG/video_ql/branch/main/graph/badge.svg?token=video_ql_token_here)](https://codecov.io/gh/AdityaNG/video_ql)
[![CI](https://github.com/AdityaNG/video_ql/actions/workflows/main.yml/badge.svg)](https://github.com/AdityaNG/video_ql/actions/workflows/main.yml)
[![GitHub License](https://img.shields.io/github/license/AdityaNG/video_ql)](https://github.com/AdityaNG/video_ql/blob/main/LICENSE)
[![PyPI - Version](https://img.shields.io/pypi/v/video_ql)](https://pypi.org/project/video_ql/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/video_ql)

Searching through video data by asking the right questions.

<img src="https://raw.githubusercontent.com/AdityaNG/video_ql/main/assets/video_ql_forklift_demo.gif">

## Install it from PyPI

```bash
pip install video_ql
```

## Usage

video_ql provides both a Python API and a CLI interface for video analysis and querying.

### Python API

```python
from video_ql import VideoQL
from video_ql.models import Query, QueryCondition, OrCondition, QueryConfig

# Define your queries
queries = [
    Query(
        query="Is the driver present in the forklift?",
        options=["yes", "no"]
    ),
    Query(
        query="Where is the forklift currently at?",
        options=["Truck", "Warehouse", "Charging"]
    ),
    Query(
        query="Is the forklift currently carrying cargo?",
        options=["yes", "no"]
    )
]

# Initialize VideoQL
video_ql = VideoQL(
    video_path="path/to/your/video.mp4",
    queries=queries,
    context="You are viewing the POV from inside a forklift"
)

# Analyze entire video
results = video_ql.analyze_video(display=True)

# Query specific conditions using our Pydantic models
query_config = QueryConfig(
    queries=[
        QueryCondition(
            query="Is the driver present in the forklift?",
            options=["yes"]
        )
    ]
)

# Query the video
matching_frames = video_ql.query_video(query_config)
```

### Command Line Interface

1. Create a config file (`config.yaml`):
```yaml
queries:
  - query: "Is the driver present in the forklift?"
    options: ["yes", "no"]
  - query: "Where is the forklift currently at?"
    options: ["Truck", "Warehouse", "Charging"]
  - query: "Is the forklift currently carrying cargo?"
    options: ["yes", "no"]
context: "You are viewing the POV from inside a forklift"
fps: 1.0
tile_frames: [3, 3]
frame_stride: 9
max_resolution: [640, 360]
```

2. Create a query file (`query.yaml`):
```yaml
queries:
  - OR:
    - query: "Is the driver present in the forklift?"
      options: ["yes"]
```

3. Run the CLI:
```bash
video_ql --video path/to/video.mp4 \
         --config config.yaml \
         --query query.yaml \
         --output results/query_results \
         --display
```

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.
