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

You can also have VideoQL automatically generate the queries and query config as shown below:
```py
from video_ql import VideoQL
from video_ql.query_proposer import generate_queries_from_context

# Define the context for your video analysis
context = "You are watching a construction site for safety compliance monitoring."

# Automatically generate queries based on the context using the selected model
queries = generate_queries_from_context(
    context=context,
    model_name="gpt-4o-mini",  # Model can be substituted as desired
    num_queries=5
)

# Initialize VideoQL with generated queries
video_ql = VideoQL(
    video_path="path/to/your/video.mp4",
    queries=queries,
    context=context
)

# Proceed with video analysis as usual
results = video_ql.analyze_video(display=True)
```

### Command Line Interface

#### Natural Language Analysis

Use the CLI tool to analyze your video
```bash
$video_ql --video path/to/video.mp4
=====================================
   Welcome to Interactive VideoQL   
=====================================

First, let's create a context for your video analysis.
Describe the video content and what you're interested in tracking or analyzing.
Video context:  ... your context here

Generating queries based on your description...

Generated queries:
1. Query 1?
   Options: Yes, No
2. Another generated query?
   Options: ...

... enjoy your video analysis
```

#### YAML Analysis
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
python3 -m video_ql.yaml_analysis --video path/to/video.mp4 \
         --config config.yaml \
         --query query.yaml \
         --output results/query_results \
         --threads 100 \
         --display
```

You may also process a single frame using the following

```bash
python3 -m video_ql.single_frame \
         --image path/to/image.png \
         --config config.yaml \
         --output results/query_results.json \
         --display
```

The query proposer helps you automatically generate relevant queries for your video content based on a provided context.
```python
from video_ql.query_proposer import generate_queries_from_context, save_queries_to_yaml

# Generate queries based on context
context = "Security camera footage of a parking lot at night"
queries = generate_queries_from_context(
    context=context,
    model_name="gpt-4o-mini",  # or "claude-3-haiku-20240307" 
    num_queries=5
)

# Save queries to a YAML file
save_queries_to_yaml(queries, "generated_queries.yaml")
```

You can use the query proposer from the command line:

```bash
python -m video_ql.query_proposer \
    --context "Dashcam footage of urban driving in rainy conditions" \
    --model "gpt-4o-mini" \
    --num-queries 7 \
    --output "dashcam_queries.yaml"
```



## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.
