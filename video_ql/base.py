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
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, create_model

from .langchain_moondream import ChatMoondream
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

            assert isinstance(query.options, list)

            if query.options:
                field_definitions[field_name] = (
                    str,
                    Field(
                        description=f"Context: {self.context}; Query: {query.query}; Choose from: {', '.join(query.options)}",  # noqa
                        default="/".join(query.options),
                    ),
                )
            else:
                field_definitions[field_name] = (
                    str,
                    Field(
                        description=query.query,
                        default="/".join(query.options),
                    ),
                )

        # Create and return the model
        FrameAnalysis = create_model(
            "FrameAnalysis", **field_definitions
        )  # type: ignore

        return FrameAnalysis

    def _create_prompt(self) -> str:
        """Create a prompt based on the queries"""
        prompt = ""  # noqa

        for query in self.queries:
            if query.options:
                prompt += f"- {query.query} Choose from: {', '.join(query.options)}\n"  # noqa
            else:
                prompt += f"- {query.query}\n"

        return prompt

    def _get_cost_per_token(self) -> Dict[str, Any]:
        """
        Get cost per token and image pricing for the current model (in USD).

        Returns a dictionary with pricing information specific to the model.
        """
        # Token pricing for different models (prompt, completion, and image)
        model_pricing = {
            # OpenAI models (as of May 2023)
            "gpt-4o-mini": {
                "prompt": 1.5e-7,  # $0.00000015 per token
                "completion": 6e-7,  # $0.0000006 per token
                "image_fixed": 0.000005,  # $0.000005 per image (fixed cost)
                "image_token_method": "fixed",  # Fixed cost per image
            },
            "gpt-4o": {
                "prompt": 5e-6,  # $0.000005 per token
                "completion": 1.5e-5,  # $0.000015 per token
                "image_fixed": 0.00002,  # $0.00002 per image (fixed cost)
                "image_token_method": "fixed",  # Fixed cost per image
            },
            "gpt-4-vision-preview": {
                "prompt": 1e-5,  # $0.00001 per token
                "completion": 3e-5,  # $0.00003 per token
                "image_fixed": 0.00002,  # $0.00002 per image (fixed cost)
                "image_token_method": "fixed",  # Fixed cost per image
            },
            # Anthropic models (as of May 2023)
            "claude-3-haiku-20240307": {
                "prompt": 2.5e-7,  # $0.00000025 per token
                "completion": 1.25e-6,  # $0.00000125 per token
                "image_token_method": "pixel_area",  # Based on pixel area
                "image_token_ratio": 750,  # pixels per token
            },
            "claude-3-sonnet-20240229": {
                "prompt": 3e-6,  # $0.000003 per token
                "completion": 1.5e-5,  # $0.000015 per token
                "image_token_method": "pixel_area",  # Based on pixel area
                "image_token_ratio": 750,  # pixels per token
            },
            "claude-3-opus-20240229": {
                "prompt": 1e-5,  # $0.00001 per token
                "completion": 6e-5,  # $0.00006 per token
                "image_token_method": "pixel_area",  # Based on pixel area
                "image_token_ratio": 750,  # pixels per token
            },
            # Google Gemini models (pricing as of July 2024)
            "gemini-2.0-flash": {
                "prompt": 1e-7,  # $0.0000001 per token ($0.1 per M)
                "completion": 7e-7,  # $0.0000007 per token ($0.7 per M)
                "image_token_method": "pixel_area",  # Based on pixel area
                "image_token_ratio": 75,  # pixels per token
            },
            "gemini-2.0-flash-lite-001": {
                "prompt": 7.5e-8,  # $0.000000075 per token ($0.075 per M)
                "completion": 3e-7,  # $0.0000003 per token ($0.3 per M)
                "image_token_method": "pixel_area",  # Based on pixel area
                "image_token_ratio": 75,  # pixels per token
            },
            "moondream-v1": {
                "prompt": 0,
                "completion": 0,
                "image_fixed": 0.0,
                "image_token_method": "fixed",
                "api_call_cost": 0.0,
            },
        }

        if self.model_name in model_pricing:
            return model_pricing[self.model_name]

        # Default pricing if model not found - conservative estimate
        return {
            "prompt": 1e-5,  # $0.00001 per token
            "completion": 3e-5,  # $0.00003 per token
            "image_fixed": 0.00002,  # $0.00002 per image (fixed cost)
            "image_token_method": "fixed",  # Fixed cost per image
        }

    def estimate_processing_cost(self) -> Dict[str, Any]:
        """
        Estimate the cost of processing the entire video.

        Returns:
            Dict containing estimated tokens and cost
        """
        # Get sample frame to estimate image size and cost
        sample_frame = self.extract_frames(0, 1)[0]
        image_cost_data = self._calculate_image_cost(sample_frame)

        # Estimate text tokens per API call (prompt instructions + completion)
        text_prompt_tokens = len(self.prompt) // 4  # ~4 chars per token
        format_instruction_tokens = 100  # Rough estimate
        avg_completion_tokens = 300  # Rough estimate for completion

        # Get pricing for the model
        pricing = self._get_cost_per_token()

        # Calculate total number of API calls needed
        total_frames_to_process = len(self)
        actual_api_calls = (
            total_frames_to_process + self.effective_stride - 1
        ) // self.effective_stride

        # For OpenAI models with fixed image cost
        if pricing["image_token_method"] == "fixed":
            # Cost per API call
            image_cost_per_call = (
                pricing.get("image_fixed", 0.00002) * self.num_frames_per_tile
            )
            text_token_cost_per_call = (
                text_prompt_tokens + format_instruction_tokens
            ) * pricing["prompt"] + avg_completion_tokens * pricing[
                "completion"
            ]
            cost_per_call = image_cost_per_call + text_token_cost_per_call

            # Calculate total estimated cost
            estimated_cost = cost_per_call * actual_api_calls

            # Estimate token usage (not including image fixed costs)
            prompt_tokens = (
                text_prompt_tokens + format_instruction_tokens
            ) * actual_api_calls
            completion_tokens = avg_completion_tokens * actual_api_calls
            total_tokens = prompt_tokens + completion_tokens

        # For Anthropic models with pixel-based image tokens
        else:
            # Image tokens per API call (for all frames in the tile)
            image_tokens_per_tile = (
                image_cost_data.get("image_tokens", 0)
                * self.num_frames_per_tile
            )

            # Total tokens per API call
            prompt_tokens_per_call = (
                text_prompt_tokens
                + format_instruction_tokens
                + image_tokens_per_tile
            )
            completion_tokens_per_call = avg_completion_tokens
            # tokens_per_call = (
            #     prompt_tokens_per_call + completion_tokens_per_call
            # )

            # Cost per API call
            cost_per_call = (prompt_tokens_per_call * pricing["prompt"]) + (
                completion_tokens_per_call * pricing["completion"]
            )

            # Calculate total estimated cost and tokens
            estimated_cost = cost_per_call * actual_api_calls
            prompt_tokens = prompt_tokens_per_call * actual_api_calls
            completion_tokens = completion_tokens_per_call * actual_api_calls
            total_tokens = prompt_tokens + completion_tokens

        return {
            "total_frames": total_frames_to_process,
            "api_calls": actual_api_calls,
            "estimated_tokens": total_tokens,
            "estimated_prompt_tokens": prompt_tokens,
            "estimated_completion_tokens": completion_tokens,
            "estimated_cost_usd": estimated_cost,
            "model": self.model_name,
            "cost_per_api_call": cost_per_call,
            "frames_per_api_call": self.num_frames_per_tile,
            "pricing_method": pricing["image_token_method"],
            "sample_image_dimensions": f"{image_cost_data['image_width']}x{image_cost_data['image_height']}",  # noqa
        }

    def _calculate_image_cost(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate the token usage and cost for processing an image based
        on model-specific pricing.

        Args:
            frame: Dictionary containing the frame data including the image

        Returns:
            Dictionary with token and cost estimates for the image
        """
        pricing = self._get_cost_per_token()
        image = frame["frame"]
        height, width = image.shape[:2]

        # Calculate based on pricing method
        if pricing["image_token_method"] == "fixed":
            # Fixed cost per image (OpenAI style)
            image_tokens = 0  # Not calculated in tokens
            image_cost = pricing.get(
                "image_fixed", 0.00002
            )  # Default to $0.00002 per image

            return {
                "image_tokens": image_tokens,
                "image_width": width,
                "image_height": height,
                "image_cost": image_cost,
            }

        elif pricing["image_token_method"] == "pixel_area":
            # Cost based on pixel area (Anthropic style)
            pixel_area = width * height
            token_ratio = pricing.get(
                "image_token_ratio", 750
            )  # Default to 750 pixels per token

            # Calculate tokens based on pixel area
            image_tokens = pixel_area / token_ratio
            image_cost = (
                image_tokens * pricing["prompt"]
            )  # Image tokens are charged at prompt rate

            return {
                "image_tokens": image_tokens,
                "image_width": width,
                "image_height": height,
                "image_cost": image_cost,
            }

        # Fallback for unknown methods
        return {
            "image_tokens": 0,
            "image_width": width,
            "image_height": height,
            "image_cost": 0,
        }

    def _analyze_frame(self, frame: Dict[str, Any]) -> Label:
        """Analyze a single frame using the vision model"""
        image_base64 = encode_image(frame["frame"])

        # Calculate image token usage and cost
        image_cost_data = self._calculate_image_cost(frame)

        if self.model_name.startswith("gpt-"):
            model = ChatOpenAI(  # type: ignore
                temperature=0.3, model=self.model_name, max_tokens=1024
            )  # type: ignore
        elif self.model_name.startswith("claude-"):
            model = ChatAnthropic(  # type: ignore
                temperature=0.3, model=self.model_name, max_tokens=1024
            )  # type: ignore
        elif self.model_name.startswith("gemini-"):
            model = ChatGoogleGenerativeAI(  # type: ignore
                temperature=0.3,
                model=self.model_name,
                max_tokens=1024,
                convert_system_message_to_human=True,  # Gemini doesn't support system messages natively  # noqa
            )  # type: ignore
        elif self.model_name.startswith("moondream-"):
            model = ChatMoondream(  # type: ignore
                api_key=os.environ.get(
                    "MOONDREAM_API_KEY"
                ),  # Will be None if not set, and the class will check env
                temperature=0.3,
                timeout=30,  # Add a reasonable timeout
            )  # type: ignore
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

        try:
            # format_instructions = self.parser.get_format_instructions()
            schema_str = self.frame_model().model_dump_json()  # type: ignore
            format_instructions = f"Conform to this json format: {schema_str}"

            messages = [
                HumanMessage(
                    content=[
                        {"type": "text", "text": self.prompt},
                        {
                            "type": "text",
                            "text": format_instructions,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ]
                )
            ]

            # Invoke the model and get response
            response = model.invoke(messages)

            # Get token usage from response if available
            token_usage = getattr(response, "usage", None)
            prompt_tokens = getattr(token_usage, "prompt_tokens", None)
            completion_tokens = getattr(token_usage, "completion_tokens", None)
            total_tokens = getattr(token_usage, "total_tokens", None)

            # If token usage isn't available in response, estimate based
            # on response length
            if prompt_tokens is None and completion_tokens is None:
                # Estimate text tokens from prompt (very rough estimate)
                text_prompt_tokens = (
                    len(self.prompt) // 4
                )  # ~4 chars per token
                format_instruction_tokens = 100  # Rough estimate

                # Use image token calculation from earlier
                image_tokens = image_cost_data.get("image_tokens", 0)

                # For models with fixed image cost, we still need
                # to estimate tokens for calculations
                if image_tokens == 0 and self.model_name.startswith("gpt-"):
                    # Very rough estimate: typical images might be ~1000 tokens
                    image_tokens = 1000

                # Total prompt tokens
                prompt_tokens = (
                    text_prompt_tokens
                    + format_instruction_tokens
                    + image_tokens
                )

                # Estimate completion tokens from response
                completion_tokens = (
                    len(response.content) // 4
                )  # ~4 chars per token

                # Total tokens
                total_tokens = prompt_tokens + completion_tokens

            # Calculate cost based on token usage and model pricing
            pricing = self._get_cost_per_token()

            # Calculate text token cost
            text_token_cost = 0
            if prompt_tokens is not None and completion_tokens is not None:
                text_token_cost = (prompt_tokens * pricing["prompt"]) + (
                    completion_tokens * pricing["completion"]
                )

            # For OpenAI, add fixed image cost
            if pricing["image_token_method"] == "fixed":
                total_cost = text_token_cost + image_cost_data["image_cost"]
            else:
                # For Anthropic, image cost is already included in
                # the prompt tokens
                total_cost = text_token_cost

            # Parse the output
            parsed_output = self.parser.parse(response.content)  # type: ignore

            # Make sure timestamp is included
            if (
                "timestamp" not in parsed_output
                or parsed_output["timestamp"] != frame["timestamp"]
            ):
                parsed_output["timestamp"] = frame["timestamp"]

            return Label(
                timestamp=frame["timestamp"],
                results=parsed_output,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost=total_cost,
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            return Label(
                timestamp=frame["timestamp"],
                results={},
                error=str(e),
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                cost=None,
            )

    def get_processing_cost(self) -> Dict[str, Any]:
        """
        Calculate the actual processing cost based on cached results.

        Returns:
            Dict containing token usage and cost information
        """
        if not self.__cache:
            return {
                "processed_frames": 0,
                "api_calls": 0,
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_cost_usd": 0,
                "model": self.model_name,
                "status": "No frames processed yet",
            }

        # Count frames with token information
        frames_with_tokens = [
            f for f in self.__cache.values() if f.total_tokens is not None
        ]
        total_api_calls = len(frames_with_tokens)

        # Calculate total tokens and cost
        total_prompt_tokens = sum(
            f.prompt_tokens or 0 for f in self.__cache.values()
        )
        total_completion_tokens = sum(
            f.completion_tokens or 0 for f in self.__cache.values()
        )
        total_tokens = sum(f.total_tokens or 0 for f in self.__cache.values())

        # Calculate cost
        pricing = self._get_cost_per_token()
        total_cost = sum(f.cost or 0 for f in self.__cache.values())

        # If cost data isn't available in the cache, calculate it
        if total_cost == 0 and total_tokens > 0:
            total_cost = (total_prompt_tokens * pricing["prompt"]) + (
                total_completion_tokens * pricing["completion"]
            )

        # Count effective frames processed (accounting for stride)
        processed_frames = sum(
            len(
                range(
                    k * self.effective_stride,
                    min((k + 1) * self.effective_stride, len(self)),
                )
            )
            for k in self.__cache.keys()
        )

        return {
            "processed_frames": processed_frames,
            "total_frames": len(self),
            "api_calls": total_api_calls,
            "total_tokens": total_tokens,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_cost_usd": total_cost,
            "model": self.model_name,
            "token_pricing": pricing,
            "completion_percentage": (
                (processed_frames / len(self)) * 100 if len(self) > 0 else 0
            ),
        }

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
