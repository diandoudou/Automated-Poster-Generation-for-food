import uuid
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

from api.models.schemas import (
    PosterStatus,
    TaskStatusResponse,
    PosterResultResponse
)

from workflow.graph import PosterState

logger = logging.getLogger(__name__)

class TaskManager:
    """
    Manages poster generation tasks.
    In production, use Redis or a database instead of in-memory storage.
    """
    def __init__(self):
        # In-memory storage
        self.tasks: Dict[str, Dict] = {}

    def create_task(self, user_input: str, max_iterations: int) -> str:
        """Create a new task"""
        task_id = str(uuid.uuid4())

        self.tasks[task_id] = {
            "task_id": task_id,
            "status": PosterStatus.PENDING,
            "progress": 0.0,
            "current_step": None,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "completed_at": None,
            "error": None,
            "poster_path": None,
            "quality_report_path": None,
            "poster_state": None,  # Will store your PosterState here
        }

        logger.info(f"Created task {task_id}")
        return task_id

    def update_task_status(
            self,
            task_id: str,
            status: PosterStatus,
            current_step: Optional[str] = None,
            progress: Optional[float] = None,
            error: Optional[str] = None
    ):
        """Update task status"""
        if task_id not in self.tasks:
            logger.warning(f"Task {task_id} not found for update")
            return

        task = self.tasks[task_id]
        task["status"] = status
        task["updated_at"] = datetime.utcnow()

        if current_step:
            task["current_step"] = current_step

        if progress is not None:
            task["progress"] = min(progress, 100.0)

        if error:
            task["error"] = error

        logger.debug(f"Task {task_id} updated: {status} - {current_step} - {progress}%")

    def complete_task(
            self,
            task_id: str,
            poster_path: Path,
            quality_report_path: Optional[str],
            result_state: PosterState
    ):
        """Mark task as completed"""
        if task_id not in self.tasks:
            logger.warning(f"Task {task_id} not found for completion")
            return

        task = self.tasks[task_id]
        task.update({
            "status": PosterStatus.COMPLETED,
            "progress": 100.0,
            "current_step": "Completed",
            "completed_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "poster_path": poster_path,
            "quality_report_path": quality_report_path,
            "poster_state": result_state  # Store the complete state
        })

        logger.info(f"Task {task_id} completed")

    def get_task_status(self, task_id: str) -> Optional[TaskStatusResponse]:
        """Get current task status"""
        if task_id not in self.tasks:
            return None

        task = self.tasks[task_id]
        return TaskStatusResponse(
            task_id=task["task_id"],
            status=task["status"],
            progress=task["progress"],
            current_step=task["current_step"],
            created_at=task["created_at"],
            updated_at=task["updated_at"],
            error=task["error"]
        )

    def get_task_result(self, task_id: str) -> Optional[PosterResultResponse]:
        """Get completed task result"""
        if task_id not in self.tasks:
            return None

        task = self.tasks[task_id]

        if task["status"] != PosterStatus.COMPLETED:
            return self.get_task_status(task_id)

        processing_time = (task["completed_at"] - task["created_at"]).total_seconds()

        # Generate URLs for outputs
        poster_url = f"/outputs/posters/{task['poster_path'].name}"
        quality_report_url = None
        if task["quality_report_path"]:
            report_path = Path(task["quality_report_path"])
            quality_report_url = f"/outputs/quality_reports/{report_path.parent.name}/{report_path.name}"

        return PosterResultResponse(
            task_id=task_id,
            status=task["status"],
            poster_url=poster_url,
            quality_report_url=quality_report_url,
            processing_time=processing_time,
            created_at=task["created_at"],
            completed_at=task["completed_at"]
        )

    def get_poster_state(self, task_id: str) -> Optional[PosterState]:
        """Get the full PosterState for a task"""
        if task_id in self.tasks:
            return self.tasks[task_id].get("poster_state")
        return None

    def get_poster_file_path(self, task_id: str) -> Optional[Path]:
        """Get the poster file path"""
        if task_id in self.tasks:
            return self.tasks[task_id].get("poster_path")
        return None

    def list_tasks(self, limit: int = 20, offset: int = 0):
        """List all tasks"""
        tasks = list(self.tasks.values())
        tasks.sort(key=lambda x: x["created_at"], reverse=True)

        return [
            TaskStatusResponse(
                task_id=t["task_id"],
                status=t["status"],
                progress=t["progress"],
                current_step=t["current_step"],
                created_at=t["created_at"],
                updated_at=t["updated_at"],
                error=t["error"]
            )
            for t in tasks[offset:offset + limit]
        ]

    def delete_task(self, task_id: str) -> bool:
        """Delete a task and its associated files"""
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]

        # Delete poster file
        if task["poster_path"] and task["poster_path"].exists():
            try:
                task["poster_path"].unlink()
                logger.info(f"Deleted poster file: {task['poster_path']}")
            except Exception as e:
                logger.error(f"Error deleting poster file: {e}")

        # Remove from memory
        del self.tasks[task_id]
        logger.info(f"Deleted task {task_id}")
        return True