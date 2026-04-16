import uvicorn
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import poster, health

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown events"""
    print("🚀 Starting Poster Generation API...")
    # Ensure output directories exist
    Path("outputs/posters").mkdir(parents=True, exist_ok=True)
    Path("outputs/quality_reports").mkdir(parents=True, exist_ok=True)
    yield
    print("👋 Shutting down...")

app = FastAPI(
    title="AI Poster Generation API",
    description="LangGraph-based multi-agent poster generation system",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated posters
outputs_dir = Path(__file__).parent.parent / "outputs"
app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")

# Routes
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(poster.router, prefix="/api/v1/poster", tags=["Poster"])

@app.get("/")
async def root():
    return {
        "name": "AI Poster Generation API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )