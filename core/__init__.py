"""
Core module for podcast search repository.

Contains the fundamental components for video encoding, embeddings, search, storage, and transcript processing.
"""

from .transcript_processor import TranscriptProcessor
from .models import Episode, Chunk
from .search_api import app

__version__ = "0.1.0"