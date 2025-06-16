"""
Core module for podcast search repository.

Contains the fundamental components for video encoding, embeddings, search, storage, and transcript processing.
"""

from .transcript_processor import TranscriptProcessor
from .models import Episode, Chunk
from .search_api import app, SearchRequest, SearchResponse, SearchResult
from .batch_processor import (
    ProcessingConfig, MemoryMonitor, StreamingProcessor,
    DeltaTracker, IncrementalProcessor, ResilientProcessor,
    ProgressTracker, ReportGenerator, MemvidBatchProcessor
)

__version__ = "0.1.0"

__all__ = [
    'app', 'SearchRequest', 'SearchResponse', 'SearchResult',
    'ProcessingConfig', 'MemoryMonitor', 'StreamingProcessor',
    'DeltaTracker', 'IncrementalProcessor', 'ResilientProcessor', 
    'ProgressTracker', 'ReportGenerator', 'MemvidBatchProcessor'
]