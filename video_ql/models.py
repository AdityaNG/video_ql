"""
Data models for video_ql project.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel


class Query(BaseModel):
    """Represents a query for video analysis"""

    query: str
    options: Optional[List[str]] = None
    short_question: Optional[str] = None
    short_options: Optional[List[str]] = None


class VideoProcessorConfig(BaseModel):
    """Configuration for video processing"""

    fps: float = 0.1
    tile_frames: Tuple[int, int] = (3, 3)
    frame_stride: int = 9
    max_resolution: Tuple[int, int] = (480, 270)
    context: str = "Answer the following"


class Label(BaseModel):
    """Represents a label/analysis result for a frame"""

    timestamp: float
    results: Dict[str, Any]
    error: Optional[str] = None

    # Costing
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost: Optional[float] = None


class QueryCondition(BaseModel):
    """Represents a single query condition"""

    query: str
    options: Optional[List[str]] = None


class AndCondition(BaseModel):
    """Represents a logical AND of multiple query conditions"""

    AND: List[QueryCondition]


class OrCondition(BaseModel):
    """Represents a logical OR of multiple query conditions"""

    OR: List[Union[QueryCondition, AndCondition]]


class QueryConfig(BaseModel):
    """Represents a complete query configuration"""

    queries: List[Union[QueryCondition, AndCondition, OrCondition]]
