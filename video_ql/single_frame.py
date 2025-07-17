"""CLI interface for analyzing single images with video_ql."""

import argparse
import json
import os

import cv2
import yaml

from .base import VideoQL
from .models import Query, VideoProcessorConfig


class SingleFrameAnalyzer:
    """A simplified version of VideoQL for analyzing single images."""

    def __init__(
        self,
        image_path,
        queries,
        context="Answer the following",
        model_name="gpt-4o-mini",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_path = image_path
        self.queries = queries
        self.context = context
        self.model_name = model_name

        # Create a temporary VideoQL instance with a dummy video
        # We'll only use its analysis capabilities
        self.ql = VideoQL(
            video_path=image_path,  # This won't be used directly
            queries=queries,
            context=context,
            video_processor_config=VideoProcessorConfig(context=context),
            disable_cache=True,  # No caching for single image
            model_name=model_name,
        )

    def analyze_image(self):
        """Analyze a single image and return the results."""
        # Load the image
        image = cv2.imread(self.image_path)
        if image is None:
            raise ValueError(f"Could not open image file: {self.image_path}")

        # Create a frame dict as expected by VideoQL
        frame = {
            "frame": image,
            "timestamp": 0.0,  # No timestamp for single image
        }

        # Use VideoQL's frame analysis method
        analysis = self.ql._analyze_frame(frame)
        return analysis


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze a single image with video_ql"
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path to the image file"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/image_analysis.json",
        help="Path to output JSON file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for analysis (e.g., gpt-4o-mini, claude-3-haiku-20240307, moondream-v1)",  # noqa
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the analyzed image with results",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    args = parse_args()
    config = load_config(args.config)

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

    # Initialize the analyzer
    analyzer = SingleFrameAnalyzer(
        image_path=args.image,
        queries=queries,
        context=config.get("context", "Answer the following"),
        model_name=args.model,
    )

    # Analyze the image
    try:
        analysis = analyzer.analyze_image()

        # Create output directory if it doesn't exist
        os.makedirs(
            os.path.dirname(os.path.abspath(args.output)), exist_ok=True
        )

        # Save analysis results
        with open(args.output, "w") as f:
            json.dump(analysis.model_dump(), f, indent=2)

        print(f"Analysis results saved to {args.output}")

        # Display results if requested
        if args.display:
            image = cv2.imread(args.image)
            vis_image = analyzer.ql._visualize_results(image, analysis)

            cv2.imshow("Image Analysis", vis_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Also save the visualization
            vis_path = os.path.splitext(args.output)[0] + "_visualization.jpg"
            cv2.imwrite(vis_path, vis_image)
            print(f"Visualization saved to {vis_path}")

    except Exception as e:
        print(f"Error analyzing image: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
