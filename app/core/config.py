from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Mineral Classifier"

    # Пути к файлам
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    MODEL_PATH: Path = BASE_DIR / "models"
    DATA_PATH: Path = BASE_DIR / "data" / "minerals.xlsx"

    class Config:
        case_sensitive = True


settings = Settings()
