import logging
from typing import List
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from api.models.schemas import (
    PosterRequest,
    PosterResponse,
    TaskStatusResponse,
    PosterResultResponse,
    PosterStatus
)
from api.services.task_manager import TaskManager
from api.services.poster_service import PosterGenerationService

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
task_manager = TaskManager()
poster_service = PosterGenerationService(task_manager)

@router.post("/generate", response_model=PosterResponse, status_code=202)
async def generate_poster(
        request: PosterRequest,
        background_tasks: BackgroundTasks
):
    """
    Start a new poster generation task using the LangGraph workflow.
    Returns immediately with a task_id for tracking progress.
    """
    try:
        # Create task
        task_id = task_manager.create_task(
            user_input=request.user_input,
            max_iterations=request.max_iterations
        )

        logger.info(f"Created task {task_id} with input: {request.user_input[:50]}...")

        # Add generation to background tasks
        background_tasks.add_task(
            poster_service.generate_poster_async,
            task_id=task_id,
            user_input=request.user_input,
            max_iterations=request.max_iterations
        )

        # Return initial response
        return PosterResponse(
            task_id=task_id,
            status=PosterStatus.PENDING,
            message="Poster generation started",
            created_at=datetime.utcnow(),
            estimated_time=120  # 2 minutes estimate
        )

    except Exception as e:
        logger.error(f"Error creating generation task: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start generation: {str(e)}")

@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Get the current status and progress of a poster generation task.
    """
    try:
        status = task_manager.get_task_status(task_id)

        if not status:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        return status

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@router.get("/result/{task_id}", response_model=PosterResultResponse)
async def get_poster_result(task_id: str):
    """
    Get the final result of a completed poster generation task.
    Includes the poster URL and quality report.
    """
    try:
        result = task_manager.get_task_result(task_id)

        if not result:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        if result.status != PosterStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail=f"Task not completed yet. Current status: {result.status}"
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task result: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get result: {str(e)}")

@router.get("/download/{task_id}")
async def download_poster(task_id: str):
    """
    Download the generated poster image file.
    """
    try:
        file_path = task_manager.get_poster_file_path(task_id)

        if not file_path or not file_path.exists():
            raise HTTPException(status_code=404, detail="Poster file not found")

        return FileResponse(
            path=str(file_path),
            media_type="image/png",
            filename=f"poster_{task_id}.png"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading poster: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to download: {str(e)}")

@router.get("/preview/{task_id}")
async def preview_poster(task_id: str):
    """
    Display the poster image directly in browser (inline).
    """
    try:
        file_path = task_manager.get_poster_file_path(task_id)

        if not file_path or not file_path.exists():
            raise HTTPException(status_code=404, detail="Poster file not found")

        return FileResponse(
            path=str(file_path),
            media_type="image/png"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error previewing poster: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/state/{task_id}")
async def get_poster_state(task_id: str):
    """
    Get the full PosterState for debugging or advanced use.
    Returns your complete workflow state.
    """
    try:
        state = task_manager.get_poster_state(task_id)

        if not state:
            raise HTTPException(status_code=404, detail="Task not found or not completed")

        return {
            "task_id": task_id,
            "poster_state": state
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting poster state: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list", response_model=List[TaskStatusResponse])
async def list_tasks(limit: int = 20, offset: int = 0):
    """
    List all poster generation tasks (most recent first).
    """
    try:
        tasks = task_manager.list_tasks(limit=limit, offset=offset)
        return tasks

    except Exception as e:
        logger.error(f"Error listing tasks: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list tasks: {str(e)}")

@router.delete("/{task_id}")
async def delete_task(task_id: str):
    """
    Delete a task and its associated files.
    """
    try:
        success = task_manager.delete_task(task_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        return {"message": f"Task {task_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting task: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete task: {str(e)}")