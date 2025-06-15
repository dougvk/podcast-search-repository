"""Logging setup."""
from loguru import logger
import sys

# Configure loguru
logger.remove()
logger.add(sys.stdout, format="<green>{time}</green> | <level>{level}</level> | {message}")
logger.add("logs/app.log", rotation="10 MB")

# Export logger
__all__ = ["logger"]