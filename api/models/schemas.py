from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional, TypedDict, Dict, List, Any

class PosterState(TypedDict):
    """共享状态"""
    user_input: str
    expanded_concept: Dict[str, Any]
    visual_output: Dict[str, Any]
    layout_output : Dict[str, Any]
    quality_evaluation: Dict[str, Any]
    quality_report_path: str
    quality_suggestions: List[str]
    next_node: Dict[str, Any]
    iteration_count: int
    max_iterations: int
    timings: Dict[str, float]  # Maps agent_name -> seconds

class PosterStatus(str, Enum):
    """Status of poster generation task"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# Request models
class PosterRequest(BaseModel):
    """API request to generate a poster"""
    user_input: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Description of the poster to generate",
        examples=["Create a seafood noodle poster with warm tones"]
    )
    max_iterations: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum refinement iterations"
    )

# Response models
class PosterResponse(BaseModel):
    """API response when generation starts"""
    task_id: str
    status: PosterStatus
    message: str
    created_at: datetime
    estimated_time: int = Field(description="Estimated completion time in seconds")

class TaskStatusResponse(BaseModel):
    """API response for task status"""
    task_id: str
    status: PosterStatus
    progress: float = Field(ge=0, le=100)
    current_step: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    error: Optional[str] = None

class PosterResultResponse(BaseModel):
    """API response for completed poster"""
    task_id: str
    status: PosterStatus
    poster_url: str  # URL to access the image
    quality_report_url: Optional[str] = None
    processing_time: float
    created_at: datetime
    completed_at: datetime