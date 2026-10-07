from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
    RAW_DATA_PATH: Path = PROJECT_ROOT / "data" / "civil_code.pdf"
    RAW_CORPUS_PATH: Path = PROJECT_ROOT / "data" / "raw_corpus.json"
    PROCESSED_CORPUS_PATH: Path = PROJECT_ROOT / "data" / "corpus.json"
    LOG_PATH: Path = PROJECT_ROOT / "logs" / "corpus_build.log"
    OCR_ZOOM_FACTOR: float = 2
    REPEALED_TEXT_AR: str = "(هذه المادة ملغاه)"
    REPEALED_TEXT_EN: str = "(this article has been repealed)"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
