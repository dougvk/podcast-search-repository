"""
API module for podcast search repository.

Contains FastAPI endpoints and request/response models.
"""

from .search_api import app
from .models import SearchRequest, SearchResponse, EpisodeInfo
from .middleware import setup_middleware

__all__ = [
    "app",
    "SearchRequest",
    "SearchResponse", 
    "EpisodeInfo",
    "setup_middleware"
]