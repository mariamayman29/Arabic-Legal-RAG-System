import logging
from pathlib import Path

from legal_rag.utils.config import settings


def configure_logging(
    log_path: Path = settings.LOG_PATH, level: int = logging.INFO
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
