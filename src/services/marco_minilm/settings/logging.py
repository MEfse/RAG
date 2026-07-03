import logging
from pathlib import Path

LOG_LEVEL = 'INFO'
PATH_LOGS = 'logging.txt'

def setup_logging() -> None:
    handlers = [logging.StreamHandler()]

    if PATH_LOGS:
        log_path = Path(PATH_LOGS)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        handlers.append(
            logging.FileHandler(PATH_LOGS, encoding="utf-8")
        )

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)