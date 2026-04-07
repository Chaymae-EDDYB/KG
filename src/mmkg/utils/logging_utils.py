import sys
from loguru import logger as _logger

def get_logger(name: str):
    return _logger.bind(module=name)

def configure_logging(level: str = "INFO") -> None:
    _logger.remove()
    _logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{extra[module]}</cyan> — {message}"
        ),
        colorize=True,
    )
