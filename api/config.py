from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    OUTPUT_DIR: Path = BASE_DIR / "outputs"
    POSTER_DIR: Path = OUTPUT_DIR / "posters"
    REPORTS_DIR: Path = OUTPUT_DIR / "quality_reports"

    # Generation Settings
    MAX_CONCURRENT_TASKS: int = 5
    TASK_TIMEOUT: int = 600  # 10 minutes
    DEFAULT_MAX_ITERATIONS: int = 3

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()