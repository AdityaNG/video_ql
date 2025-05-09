"""CLI interface for video_ql project."""

import argparse
import json
import os

import cv2
import yaml

from .base import VideoQL
from .models import Query, QueryConfig, VideoProcessorConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Query video analysis results"
    )
    parser.add_argument(
        "--video", type=str, required=True, help="Path to the video file"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to the results JSON file",
    )
    parser.add_argument(
        "--query", type=str, required=True, help="Path to the query YAML file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/query_results",
        help="Path to output directory",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display frames that match the query",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():  # pragma: no cover
    args = parse_args()
    config = load_config(args.config)

    video_processor_config = VideoProcessorConfig(
        **{
            "fps": config.get("fps"),
            "tile_frames": config.get("tile_frames"),
            "frame_stride": config.get("frame_stride"),
            "max_resolution": config.get("max_resolution"),
        }
    )

    # Convert config queries to Query objects
    queries = [
        Query(
            query=q["query"],
            options=q.get("options"),
            short_query=q.get("short_query"),
            short_options=q.get("short_options"),
        )
        for q in config["queries"]
    ]

    # Initialize VideoQL
    video_ql = VideoQL(
        video_path=args.video,
        queries=queries,
        context=config.get("context", "Answer the following"),
        # The rest of the config options
        video_processor_config=video_processor_config,
    )

    # Load the query
    query_data = load_config(args.query)
    query_config = QueryConfig(**query_data)

    # Find matching frames
    matching_frames = video_ql.query_video(query_config)

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Save and/or display matching frames
    if matching_frames:
        print(f"Found {len(matching_frames)} matching frames")

        # Save information about matching frames
        with open(os.path.join(args.output, "query_results.json"), "w") as f:
            json.dump(
                {
                    "query_config": query_config.dict(),
                    "matching_frames": [
                        {
                            "index": idx,
                            "timestamp": video_ql[
                                idx // config.get("frame_stride", 9)
                            ].timestamp,
                        }
                        for idx in matching_frames
                    ],
                },
                f,
                indent=2,
            )

        # Display or save matching frames
        for i, idx in enumerate(matching_frames):
            analysis = video_ql[idx]

            # Extract the frame
            frames = video_ql.extract_frames(idx, 1)
            if frames:
                frame = frames[0]["frame"]
                vis_frame = video_ql._visualize_results(frame, analysis)

                # Save the frame
                frame_path = os.path.join(
                    args.output,
                    f"match_{i:03d}_frame_{idx:04d}_{analysis.timestamp:.2f}s.jpg",  # noqa
                )
                cv2.imwrite(frame_path, vis_frame)

                # Display if requested
                if args.display:
                    cv2.imshow("Query Results", vis_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC key
                        break

        if args.display:
            cv2.destroyAllWindows()
    else:
        print("No frames matched the query")
