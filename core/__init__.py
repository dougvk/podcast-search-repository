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
from .cache_manager import CacheManager, CacheConfig, get_cache_manager, init_cache
from .index_optimizer import IndexOptimizer, IndexConfig, IndexType, create_optimized_index
from .embedding_manager import EmbeddingManager, CompressedEmbeddings, get_embedding_manager, compress_embeddings
from .async_handler import AsyncRequestHandler, ConcurrentSearchProcessor, get_async_handler, concurrent_search
from .performance_monitor import PerformanceProfiler, BottleneckDetector, get_profiler, profile, track_endpoint_performance
from .system_tuner import SystemTuner, get_system_tuner, auto_tune_system, optimize_for_search

__version__ = "0.1.0"

__all__ = [
    'app', 'SearchRequest', 'SearchResponse', 'SearchResult',
    'ProcessingConfig', 'MemoryMonitor', 'StreamingProcessor',
    'DeltaTracker', 'IncrementalProcessor', 'ResilientProcessor', 
    'ProgressTracker', 'ReportGenerator', 'MemvidBatchProcessor',
    'CacheManager', 'CacheConfig', 'get_cache_manager', 'init_cache',
    'IndexOptimizer', 'IndexConfig', 'IndexType', 'create_optimized_index',
    'EmbeddingManager', 'CompressedEmbeddings', 'get_embedding_manager', 'compress_embeddings',
    'AsyncRequestHandler', 'ConcurrentSearchProcessor', 'get_async_handler', 'concurrent_search',
    'PerformanceProfiler', 'BottleneckDetector', 'get_profiler', 'profile', 'track_endpoint_performance',
    'SystemTuner', 'get_system_tuner', 'auto_tune_system', 'optimize_for_search'
]