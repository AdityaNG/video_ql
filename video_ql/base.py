"""
video_ql base module.
"""

import hashlib
import json
import os
import threading
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import yaml
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, create_model

from .models import Label, Query, QueryConfig, VideoProcessorConfig
from .query import matches_query
from .query_proposer import (
    generate_queries_from_context,
    generate_query_config_from_question,
)
from .utils import encode_image, get_length_of_video, get_video_fps, video_hash
from .visualization import VideoVisualizer

NAME = "video_ql"


class VideoQL:
    __cache: Dict[int, Label] = {}

    def __init__(
        self,
        video_path: str,
        queries: List[Query],
        context: str = "Answer the following",
        video_processor_config: Optional[VideoProcessorConfig] = None,
        cache_dir: str = "~/.cache/video_ql/",
        disable_cache: bool = False,
        model_name: str = "gpt-4o-mini",
        # model_name: str = "claude-3-haiku-20240307",
    ):
        """Initialize the VideoQL instance"""
        self.video_path = video_path
        self.queries = queries
        self.context = context
        self.disable_cache = disable_cache
        self.model_name = model_name

        # Expand the cache directory if it starts with ~
        self.cache_dir = os.path.expanduser(cache_dir)

        # Create default config if not provided
        if video_processor_config is None:
            self.config = VideoProcessorConfig(context=context)
        else:
            self.config = video_processor_config
            if not hasattr(self.config, "context"):
                self.config.context = context

        # Generate a unique hash for this video analysis setup
        self.scene_hash = self._generate_scene_hash()
        self.cache_path = os.path.join(
            self.cache_dir, f"{self.scene_hash}.json"
        )

        # Get video info
        self.num_video_frames = get_length_of_video(video_path)
        self.num_frames_per_tile = (
            self.config.tile_frames[0] * self.config.tile_frames[1]
        )
        self.video_fps = get_video_fps(video_path)
        # Calculate the correct frame stride based on fps adjustment
        self.effective_stride = int(
            self.config.frame_stride * (self.video_fps / self.config.fps)
        )

        # Visualizer
        self.visualizer = VideoVisualizer()

        # Create the frame analysis model
        self.frame_model = self._create_frame_model()
        self.parser = JsonOutputParser(
            pydantic_object=self.frame_model  # type: ignore
        )

        # Load or initialize the cache
        if not os.path.exists(os.path.dirname(self.cache_path)):
            os.makedirs(os.path.dirname(self.cache_path))

        # Cache lock for thread safety
        self.__cache_lock = threading.RLock()
        self.prompt = self._create_prompt()

        self._load_cache()

    def _generate_scene_hash(self) -> str:
        """Generate a unique hash for this video analysis setup"""
        # Hash the video file
        v_hash = video_hash(self.video_path)

        # Create a string representing the queries and config
        query_str = json.dumps(
            [q.model_dump() for q in self.queries], sort_keys=True
        )
        config_str = json.dumps(self.config.model_dump(), sort_keys=True)
        context_str = self.context

        # Combine all components and hash
        combined = f"{v_hash}_{query_str}_{config_str}_{context_str}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _load_cache(self) -> Dict[int, Label]:
        """Load the cache from disk"""
        cache: Dict[int, Label] = {}
        if self.disable_cache:
            return cache

        with self.__cache_lock:
            if os.path.exists(self.cache_path):
                try:
                    with open(self.cache_path, "r") as f:
                        cache_data = json.load(f)
                    for key, value in cache_data.items():
                        cache[int(key)] = Label(**value)
                except Exception as e:
                    print(f"Warning: Could not load cache: {e}")

            self.__cache = cache

        return self.__cache

    def _update_cache(
        self,
        cache_idx: int,
        analysis: Label,
    ):
        """Save the cache to disk"""
        if self.disable_cache:
            return

        with self.__cache_lock:  # Acquire lock before saving
            self.__cache[cache_idx] = analysis

            cache_data = {k: v.model_dump() for k, v in self.__cache.items()}
            with open(self.cache_path, "w") as f:
                json.dump(cache_data, f, indent=2)

        self._load_cache()

    def _create_frame_model(self) -> BaseModel:
        """Create a Pydantic model based on the queries"""
        field_definitions = {}

        for query in self.queries:
            field_name = query.query.lower().replace("?", "").replace(" ", "_")

            if query.options:
                field_definitions[field_name] = (
                    str,
                    Field(
                        description=f"Context: {self.context}; Query: {query.query}; Choose from: {', '.join(query.options)}"  # noqa
                    ),
                )
            else:
                field_definitions[field_name] = (
                    str,
                    Field(description=query.query),
                )

        # Add timestamp field
        field_definitions["timestamp"] = (
            float,
            Field(description="Timestamp of the frame in seconds"),
        )  # type: ignore

        # Create and return the model
        FrameAnalysis = create_model(
            "FrameAnalysis", **field_definitions
        )  # type: ignore

        return FrameAnalysis

    def _create_prompt(self) -> str:
        """Create a prompt based on the queries"""
        prompt = "Analyze this video frame and provide the following information:\n\n"  # noqa

        for query in self.queries:
            if query.options:
                prompt += f"- {query.query} Choose from: {', '.join(query.options)}\n"  # noqa
            else:
                prompt += f"- {query.query}\n"

        return prompt

    def _analyze_frame(self, frame: Dict[str, Any]) -> Label:
        """Analyze a single frame using the vision model"""
        image_base64 = encode_image(frame["frame"])

        if self.model_name in ["gpt-4o-mini"]:
            model = ChatOpenAI(
                temperature=0.3, model=self.model_name, max_tokens=1024
            )  # type: ignore
        elif self.model_name in ["claude-3-haiku-20240307"]:
            model = ChatAnthropic(
                temperature=0.3, model=self.model_name, max_tokens=1024
            )  # type: ignore
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

        try:
            msg = model.invoke(
                [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": self.prompt},
                            {
                                "type": "text",
                                "text": self.parser.get_format_instructions(),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"  # noqa
                                },
                            },
                        ]
                    )
                ]
            )

            # Parse the output
            parsed_output = self.parser.parse(msg.content)  # type: ignore

            # Make sure timestamp is included
            if (
                "timestamp" not in parsed_output
                or parsed_output["timestamp"] != frame["timestamp"]
            ):
                parsed_output["timestamp"] = frame["timestamp"]

            return Label(timestamp=frame["timestamp"], results=parsed_output)

        except Exception as e:
            return Label(
                timestamp=frame["timestamp"], results={}, error=str(e)
            )

    def __len__(self) -> int:
        """Return the number of frames that can be processed"""
        stride = self.config.frame_stride * (self.video_fps / self.config.fps)
        return max(0, int(self.num_video_frames - stride))

    def __getitem__(self, idx: int) -> Label:
        """Get the analysis for a specific frame index"""
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of bounds for video with {len(self)} frames"
            )

        # Calculate the cache index using the effective stride
        cache_idx = idx // self.effective_stride

        if cache_idx not in self.__cache:
            # Calculate the actual frame index in the video
            frame_idx = cache_idx * self.effective_stride

            # Extract frames for the tile
            total_frames_needed = (
                self.config.tile_frames[0] * self.config.tile_frames[1]
            )
            frames = self.extract_frames(
                frame_idx, total_frames_needed, self.config.frame_stride
            )

            # Create a tile image from these frames
            tile_image = self.create_tile_image_from_frames(frames)

            # Use the timestamp from the first frame
            first_frame_timestamp = frames[0]["timestamp"] if frames else 0

            frame = {
                "frame": tile_image,
                "timestamp": first_frame_timestamp,
            }

            analysis = self._analyze_frame(frame)
            # Update the cache
            self._update_cache(cache_idx, analysis)

        return self.__cache[cache_idx]

    def get_frame_analysis(self, timestamp: float) -> Label:
        """Get the frame analysis at a specific timestamp"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = int(timestamp * fps)
        cap.release()

        return self[frame_idx]

    def analyze_video(
        self,
        display: bool = False,
        save_frames: bool = False,
        output_dir: str = "results",
    ):
        """Process the entire video and return all analysis results"""
        results = []

        # Create output directory for frames if needed
        if save_frames:
            frames_dir = os.path.join(output_dir, "frames")
            if not os.path.exists(frames_dir):
                os.makedirs(frames_dir)

        # Get total number of frames to process
        total_frames = len(self)

        for i in range(0, total_frames):
            try:
                analysis = self[i]
                results.append(analysis)

                # If we need to display or save, get the original frame
                if display or save_frames:
                    frames = self.extract_frames(i, 1)
                    if frames:
                        vis_frame = self._visualize_results(
                            frames[0]["frame"], analysis
                        )

                        if display:
                            cv2.imshow("Video Analysis", vis_frame)
                            key = cv2.waitKey(1) & 0xFF
                            if key == 27:  # ESC key
                                break

                        if save_frames:
                            frame_path = os.path.join(
                                frames_dir,
                                f"frame_{i:04d}_{analysis.timestamp:.2f}s.jpg",
                            )
                            cv2.imwrite(frame_path, vis_frame)

            except Exception as e:
                print(f"Error processing frame at index {i}: {e}")
                import traceback

                traceback.print_exc()

        # Close any open windows
        if display:
            cv2.destroyAllWindows()

        return results

    def _visualize_results(
        self, frame: np.ndarray, analysis: Label
    ) -> np.ndarray:
        """
        Overlay analysis results on the frame with a clean,
        professional look.
        """
        return self.visualizer.visualize_results(frame, analysis)

    def create_tile_image_from_frames(
        self, frames: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Create a tiled image from multiple frames."""
        return self.visualizer.create_tile_image_from_frames(
            frames, self.config.tile_frames
        )

    def extract_frames(
        self,
        start_idx: int,
        count: int,
        stride: int = 1,
    ) -> List[Dict[str, Any]]:
        frames = self.visualizer.extract_frames(
            self.video_path,
            start_idx,
            count,
            stride,
            self.config.max_resolution,
        )
        return frames

    def create_tile_image(
        self, start_idx: int = 0, stride: int = 1
    ) -> np.ndarray:
        """Create a tiled image of frames starting from start_idx"""
        frames = self.extract_frames(
            start_idx,
            self.config.tile_frames[0] * self.config.tile_frames[1],
            stride,
        )

        return self.create_tile_image_from_frames(frames)

    def query_video(
        self,
        query_config: Union[str, Dict, QueryConfig],
        output_path: str = "results/query_output.mp4",
    ) -> List[int]:
        """
        Query the video based on a query configuration
        Returns indices of frames that match the query

        Args:
            query_config: Path to a YAML file, a dict, or a QueryConfig object
                containing query configuration
            output_path: Path to save the output video
        """
        # Load query config if it's a path
        if isinstance(query_config, str):
            with open(query_config, "r") as f:
                query_data = yaml.safe_load(f)
                query_config = QueryConfig(**query_data)
        # Convert dict to QueryConfig if needed
        elif isinstance(query_config, dict):
            query_config = QueryConfig(**query_config)

        matching_frames = []

        # Process all frames first (if not already in cache)
        self.analyze_video(display=False, save_frames=False)

        # Now search through cached results
        for idx in sorted(self.__cache.keys()):
            analysis = self.__cache[idx]

            # If the analysis matches any of the queries
            if matches_query(analysis, query_config.queries):

                video_idx_lb = int(idx * self.effective_stride)
                video_idx_ub = int((idx + 1) * self.effective_stride)

                for video_idx in range(video_idx_lb, video_idx_ub):
                    matching_frames.append(video_idx)

        return matching_frames

    def generate_queries(
        self,
        context: Optional[str] = None,
        model_name: Optional[str] = None,
        num_queries: int = 5,
    ) -> List[Query]:
        """
        Generate relevant queries for the video based on a context description.

        Args:
            context: Description of the video content and analysis goals
                (defaults to self.context)
            model_name: LLM model to use (defaults to self.model_name)
            num_queries: Number of queries to generate

        Returns:
            List of Query objects
        """
        if context is None:
            context = self.context

        if model_name is None:
            model_name = self.model_name

        return generate_queries_from_context(
            context=context, model_name=model_name, num_queries=num_queries
        )

    def generate_query_config(
        self, question: str, model_name: Optional[str] = None
    ) -> QueryConfig:
        """
        Generate a QueryConfig from a natural language question.

        This converts a natural language question into a structured query
        configuration that can be used to query the video.

        Args:
            question: Natural language question about the video
            model_name: LLM model to use (defaults to self.model_name)

        Returns:
            QueryConfig object
        """
        if model_name is None:
            model_name = self.model_name

        # First make sure we have some analysis results to work with
        if not self.__cache:
            # Process at least a few frames to build some context
            for i in range(min(5, len(self))):
                self[i]  # This will compute and cache the frame analysis

        return generate_query_config_from_question(
            queries=self.queries,
            context=self.context,
            analysis=self.__cache,
            question=question,
            model_name=model_name,
        )
