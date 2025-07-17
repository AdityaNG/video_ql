"""CLI interface for video_ql project."""

import argparse
import concurrent.futures
import json
import os
from typing import List

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
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads to use for parallel processing",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def process_frame_chunk(video_ql: VideoQL, indices: List[int]):
    """Process a chunk of frames in a thread"""
    for idx in indices:
        try:
            video_ql[idx]  # This will compute and cache the frame analysis
        except Exception as e:
            print(f"Error processing frame {idx}: {e}")


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
        model_name="moondream-v1",
    )

    # Parallelize the cache population
    total_frames = len(video_ql)
    max_threads = min(
        args.threads, total_frames
    )  # Don't create more threads than frames

    print(f"Processing {total_frames} frames using {max_threads} threads...")

    # Split the frames into chunks for each thread
    if max_threads > 0:
        chunk_size = max(1, total_frames // max_threads)
        chunks = []

        for i in range(0, total_frames, chunk_size):
            end = min(i + chunk_size, total_frames)
            chunks.append(list(range(i, end)))

        # Use a thread pool to process the chunks in parallel
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_threads
        ) as executor:
            futures = []
            for chunk in chunks:
                future = executor.submit(process_frame_chunk, video_ql, chunk)
                futures.append(future)

            # Wait for all threads to complete
            for i, future in enumerate(
                concurrent.futures.as_completed(futures)
            ):
                print(f"Thread {i+1}/{len(futures)} completed")
                # Get any exceptions that occurred
                try:
                    future.result()
                except Exception as e:
                    print(f"Thread error: {e}")

    print("Frame processing complete. Running query...")

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
                    "query_config": query_config.model_dump(),
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


if __name__ == "__main__":
    main()
