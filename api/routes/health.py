from fastapi import APIRouter
from datetime import datetime
from api.config import settings

router = APIRouter()

@router.get("")
@router.get("/")
async def health_check():
    """
    Comprehensive health check of all system components.
    """
    services = {}

    # Check API
    services["api"] = "healthy"

    # Check filesystem
    try:
        settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        services["filesystem"] = "healthy"
    except Exception as e:
        services["filesystem"] = f"unhealthy: {str(e)}"

    # Check workflow
    try:
        from workflow.graph import build_workflow
        services["workflow"] = "healthy"
    except Exception as e:
        services["workflow"] = f"unhealthy: {str(e)}"

    # Check RAG
    try:
        from rag.rag import RAGSystem
        services["rag"] = "healthy"
    except Exception as e:
        services["rag"] = "unavailable"

    overall = "healthy" if all(
        s in ["healthy", "unavailable"] for s in services.values()
    ) else "degraded"

    return {
        "status": overall,
        "timestamp": datetime.utcnow(),
        "version": "1.0.0",
        "services": services
    }

@router.get("/ready")
async def readiness():
    """Kubernetes readiness probe"""
    return {"status": "ready"}

@router.get("/live")
async def liveness():
    """Kubernetes liveness probe"""
    return {"status": "alive"}