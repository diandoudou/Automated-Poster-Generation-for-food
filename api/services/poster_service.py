import logging
import asyncio
from pathlib import Path

from api.config import settings
from api.services.task_manager import TaskManager
from api.models.schemas import PosterStatus, PosterState

from workflow.graph import build_workflow

logger = logging.getLogger(__name__)

class PosterGenerationService:
    """Service that runs the LangGraph workflow for poster generation"""
    def __init__(self, task_manager: TaskManager):
        self.task_manager = task_manager
        self.workflow_graph = None

    def _get_workflow(self):
        """Lazy initialization of workflow graph"""
        if self.workflow_graph is None:
            try:
                self.workflow_graph = build_workflow().compile()
                logger.info("Workflow compiled successfully")
            except Exception as e:
                logger.error(f"Failed to compile workflow: {str(e)}")
                raise
        return self.workflow_graph

    async def generate_poster_async(
            self,
            task_id: str,
            user_input: str,
            max_iterations: int
    ):
        """
        Run the poster generation workflow asynchronously.
        This is executed as a background task.
        """
        try:
            logger.info(f"Starting poster generation for task {task_id}")

            # Update status to processing
            self.task_manager.update_task_status(
                task_id=task_id,
                status=PosterStatus.PROCESSING,
                current_step="Initializing workflow",
                progress=5.0
            )

            # Get compiled workflow
            workflow = self._get_workflow()

            # Create initial state using your PosterState structure
            init_state: PosterState = {
                "user_input": user_input,
                "iteration_count": 0,
                "max_iterations": max_iterations,
                # These will be filled by agents
                "expanded_concept": {},
                "visual_output": {},
                "layout_output": {},
                "quality_evaluation": {},
                "quality_report_path": "",
                "quality_suggestions": [],
                "next_node": {}
            }

            # Run workflow in thread (LangGraph invoke is synchronous)
            logger.info(f"Invoking workflow for task {task_id}")
            result_state = await asyncio.to_thread(workflow.invoke, init_state)
            logger.info(f"Workflow completed for task {task_id}")

            # Process the completed result
            await self._process_result(task_id, result_state, user_input)

            logger.info(f"Task {task_id} completed successfully")

        except Exception as e:
            logger.error(f"Task {task_id} failed: {str(e)}", exc_info=True)
            self.task_manager.update_task_status(
                task_id=task_id,
                status=PosterStatus.FAILED,
                error=str(e)
            )

    async def _process_result(
            self,
            task_id: str,
            result_state: PosterState,
            user_input: str
    ):
        """Process completed workflow result"""

        # Find the generated poster file
        poster_path = None

        # Check layout_output for image path
        if result_state.get("layout_output") and "layout_path" in result_state["layout_output"]:
            poster_path = Path(result_state["layout_output"]["layout_path"])
            logger.info(f"Found poster path in layout_output: {poster_path}")

        if not poster_path or not poster_path.exists():
            raise FileNotFoundError(f"Generated poster not found for task {task_id}")

        # Copy to task-specific location
        final_path = settings.POSTER_DIR / f"poster_{task_id}.png"
        import shutil
        shutil.copy(poster_path, final_path)
        logger.info(f"Copied poster to: {final_path}")

        # Get quality report path from state
        quality_report_path = result_state.get("quality_report_path")

        # Complete the task
        self.task_manager.complete_task(
            task_id=task_id,
            poster_path=final_path,
            quality_report_path=quality_report_path,
            result_state=result_state  # Pass the whole state
        )