from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=Path(__file__).parent.parent.parent / ".env", env_file_encoding="utf-8",extra="ignore")

    RAW_DATA_PATH: Path = Path("data/raw")
    PROCESSED_DATA_PATH: Path = Path("data/corpus.json")

    OCR_ZOOM_FACTOR: float = 2.0
    REPEALED_TEXT_AR: str = "(هذاالقانون ملغى)"
    REPEALED_TEXT_EN: str = "(this law has been repealed)"

settings = Settings()