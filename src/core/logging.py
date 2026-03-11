import logging
from pathlib import Path
from .config import settings


def setup_logging() -> None:
    handlers = [logging.StreamHandler()]

    if settings.path_logging:
        log_path = Path(settings.path_logging)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.FileHandler(log_path, encoding="utf-8")
        )

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )
