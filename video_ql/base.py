"""
video_ql base module.
"""

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import yaml
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, create_model

from .models import Label, Query, VideoProcessorConfig
from .query import matches_query
from .utils import encode_image, get_length_of_video, get_video_fps, video_hash
from .visualization import VideoVisualizer

NAME = "video_ql"


class VideoQL:
    def __init__(
        self,
        video_path: str,
        queries: List[Query],
        context: str = "Answer the following",
        video_processor_config: Optional[VideoProcessorConfig] = None,
        cache_dir: str = "~/.cache/video_ql/",
        disable_cache: bool = False,
        # model_name: str = "gpt-4o-mini",
        model_name: str = "claude-3-haiku-20240307",
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

        self.__cache = self._load_cache()
        self.prompt = self._create_prompt()

    def _generate_scene_hash(self) -> str:
        """Generate a unique hash for this video analysis setup"""
        # Hash the video file
        v_hash = video_hash(self.video_path)

        # Create a string representing the queries and config
        query_str = json.dumps(
            [q.dict() for q in self.queries], sort_keys=True
        )
        config_str = json.dumps(self.config.dict(), sort_keys=True)
        context_str = self.context

        # Combine all components and hash
        combined = f"{v_hash}_{query_str}_{config_str}_{context_str}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _load_cache(self) -> Dict[int, Label]:
        """Load the cache from disk"""
        cache: Dict[int, Label] = {}
        if self.disable_cache:
            return cache

        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r") as f:
                    cache_data = json.load(f)
                for key, value in cache_data.items():
                    cache[int(key)] = Label(**value)
            except Exception as e:
                print(f"Warning: Could not load cache: {e}")
        return cache

    def _save_cache(self):
        """Save the cache to disk"""
        if self.disable_cache:
            return

        cache_data = {k: v.dict() for k, v in self.__cache.items()}
        with open(self.cache_path, "w") as f:
            json.dump(cache_data, f, indent=2)

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

    def _extract_frames(
        self,
        start_idx: int,
        count: int,
        stride: int = 1,
    ) -> List[Dict[str, Any]]:
        """Extract frames from video starting at index"""
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []  # type: ignore

        # Set position to start_idx
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

        for _ in range(count):
            ret, frame = cap.read()

            if not ret:
                break

            # Resize frame if needed
            if frame is not None:
                h, w = frame.shape[:2]
                max_w, max_h = self.config.max_resolution

                scale = 1.0
                if w > max_w or h > max_h:
                    scale_w = max_w / w
                    scale_h = max_h / h
                    scale = min(scale_w, scale_h)

                if scale < 1.0:
                    new_w, new_h = int(w * scale), int(h * scale)
                    frame = cv2.resize(
                        frame, (new_w, new_h), interpolation=cv2.INTER_AREA
                    )

            timestamp = (start_idx + len(frames)) / video_fps
            frames.append({"frame": frame, "timestamp": timestamp})

            # Stride jump
            for _ in range(stride - 1):
                ret, frame = cap.read()

        cap.release()
        return frames

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
            frames = self._extract_frames(
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
            self.__cache[cache_idx] = analysis
            # Save the updated cache
            self._save_cache()

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
                print(
                    f"Processed frame at timestamp: {analysis.timestamp:.2f}s"
                )

                # If we need to display or save, get the original frame
                if display or save_frames:
                    frames = self._extract_frames(i, 1)
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

    def create_tile_image(
        self, start_idx: int = 0, stride: int = 1
    ) -> np.ndarray:
        """Create a tiled image of frames starting from start_idx"""
        frames = self._extract_frames(
            start_idx,
            self.config.tile_frames[0] * self.config.tile_frames[1],
            stride,
        )

        return self.create_tile_image_from_frames(frames)

    def query_video(
        self,
        query_config: Union[str, Dict],
        display: bool = False,
        save_video: bool = False,
        output_path: str = "results/query_output.mp4",
    ) -> List[int]:
        """
        Query the video based on a query configuration
        Returns indices of frames that match the query

        Args:
            query_config: Path to a YAML file or a dict containing query
                configuration
            display: Whether to display matching segments
            save_video: Whether to save matching segments to a video file
            output_path: Path to save the output video (if save_video is True)
        """
        # Load query config if it's a path
        if isinstance(query_config, str):
            with open(query_config, "r") as f:
                query_config = yaml.safe_load(f)

        matching_frames = []

        # Process all frames first (if not already in cache)
        self.analyze_video(display=False, save_frames=False)

        # Now search through cached results
        for idx in sorted(self.__cache.keys()):
            analysis = self.__cache[idx]

            # If the analysis matches any of the queries
            if matches_query(
                analysis, query_config["queries"]  # type: ignore
            ):

                video_idx_lb = int(idx * self.effective_stride)
                video_idx_ub = int((idx + 1) * self.effective_stride)

                for video_idx in range(video_idx_lb, video_idx_ub):
                    matching_frames.append(video_idx)

        if display or save_video:
            self.play_matching_segments(
                matching_frames,
                output_path=output_path if save_video else None,
            )

        return matching_frames

    def play_matching_segments(
        self,
        matching_frames: List[int],
        display_time: float = 0.1,
        output_path: Optional[str] = None,
    ) -> None:
        """Play video segments that match the query criteria.

        Args:
            matching_frames: List of frame indices that match the query
            display_time: Time to display each frame in seconds
            output_path: Path to save output video (optional)
        """
        if not matching_frames:
            print("No matching frames found.")
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Setup output video writer if requested
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Group consecutive frames into segments
        segments = []
        current_segment = [matching_frames[0]]

        for i in range(1, len(matching_frames)):
            if (
                matching_frames[i] - matching_frames[i - 1]
                <= self.config.frame_stride * 2
            ):
                current_segment.append(matching_frames[i])
            else:
                segments.append(current_segment)
                current_segment = [matching_frames[i]]

        segments.append(current_segment)

        print(f"Found {len(segments)} matching segments")

        # Play each segment
        for segment in segments:
            start_idx = segment[0]
            end_idx = segment[-1]

            print(f"Playing segment from frame {start_idx} to {end_idx}")

            for idx in range(start_idx, end_idx + 1, self.config.frame_stride):
                # Get the analysis for this frame
                analysis = self[idx]

                # Get the actual frame from video
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()

                if not ret:
                    break

                # Visualize the results
                vis_frame = self._visualize_results(frame, analysis)

                # Display
                cv2.imshow("Query Results", vis_frame)

                # Write to output if requested
                if writer:
                    writer.write(vis_frame)

                # Wait for key press
                key = cv2.waitKey(int(display_time * 1000)) & 0xFF
                if key == 27:  # ESC key
                    break

            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break

        # Release resources
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
