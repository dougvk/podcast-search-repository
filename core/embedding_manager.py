"""
High-Performance Embedding Compression and Lazy Loading Manager

Follows memvid patterns from MEMVID.md with minimal code for maximum efficiency.
Implements compression, lazy loading, and memory-efficient embedding management.
"""

import gzip
import pickle
import threading
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class CompressedEmbeddings:
    """Memory-efficient compressed embedding storage with lazy loading"""
    
    def __init__(self, embeddings: np.ndarray = None, compression_level: int = 6):
        self._lock = threading.RLock()
        self._compression_level = compression_level
        self._compressed_data: Optional[bytes] = None
        self._cached_embeddings: Optional[np.ndarray] = None
        self._is_loaded = False
        self._stats = {"compressions": 0, "decompressions": 0, "cache_hits": 0}
        
        if embeddings is not None:
            self.compress_embeddings(embeddings)
    
    def compress_embeddings(self, embeddings: np.ndarray) -> float:
        """Compress embeddings with gzip, return compression ratio"""
        with self._lock:
            original_bytes = embeddings.nbytes
            compressed = gzip.compress(pickle.dumps(embeddings), compresslevel=self._compression_level)
            self._compressed_data = compressed
            self._cached_embeddings = None
            self._is_loaded = False
            self._stats["compressions"] += 1
            
            ratio = original_bytes / len(compressed) if compressed else 1.0
            logger.debug(f"Compressed {original_bytes} bytes to {len(compressed)} (ratio: {ratio:.2f}x)")
            return ratio
    
    def get_embeddings(self) -> np.ndarray:
        """Lazy load and decompress embeddings with caching"""
        with self._lock:
            if self._cached_embeddings is not None:
                self._stats["cache_hits"] += 1
                return self._cached_embeddings
            
            if self._compressed_data is None:
                raise ValueError("No compressed data available")
            
            self._cached_embeddings = pickle.loads(gzip.decompress(self._compressed_data))
            self._is_loaded = True
            self._stats["decompressions"] += 1
            return self._cached_embeddings
    
    def unload(self):
        """Free cached embeddings to save memory"""
        with self._lock:
            self._cached_embeddings = None
            self._is_loaded = False
    
    def save_to_file(self, filepath: str):
        """Save compressed embeddings to disk"""
        with self._lock:
            if self._compressed_data is None:
                raise ValueError("No data to save")
            Path(filepath).write_bytes(self._compressed_data)
    
    def load_from_file(self, filepath: str):
        """Load compressed embeddings from disk"""
        with self._lock:
            self._compressed_data = Path(filepath).read_bytes()
            self._cached_embeddings = None
            self._is_loaded = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get compression and access statistics"""
        with self._lock:
            return {
                "is_loaded": self._is_loaded,
                "has_compressed_data": self._compressed_data is not None,
                "compressed_size_bytes": len(self._compressed_data) if self._compressed_data else 0,
                **self._stats
            }

class EmbeddingManager:
    """Enterprise-grade embedding manager with compression and lazy loading"""
    
    def __init__(self, compression_level: int = 6, auto_unload_threshold: int = 100):
        self._lock = threading.RLock()
        self._embeddings: Dict[str, CompressedEmbeddings] = {}
        self._compression_level = compression_level
        self._auto_unload_threshold = auto_unload_threshold
        self._access_count = 0
        self._stats = {"total_embeddings": 0, "memory_saved_mb": 0.0}
    
    def add_embeddings(self, key: str, embeddings: np.ndarray) -> float:
        """Add embeddings with automatic compression"""
        with self._lock:
            compressed_emb = CompressedEmbeddings(embeddings, self._compression_level)
            self._embeddings[key] = compressed_emb
            self._stats["total_embeddings"] += 1
            
            # Calculate memory savings
            original_mb = embeddings.nbytes / (1024 * 1024)
            compressed_mb = len(compressed_emb._compressed_data) / (1024 * 1024)
            saved_mb = original_mb - compressed_mb
            self._stats["memory_saved_mb"] += saved_mb
            
            return original_mb / compressed_mb if compressed_mb > 0 else 1.0
    
    def get_embeddings(self, key: str) -> np.ndarray:
        """Get embeddings with lazy loading and auto-unload management"""
        with self._lock:
            if key not in self._embeddings:
                raise KeyError(f"Embeddings not found: {key}")
            
            embeddings = self._embeddings[key].get_embeddings()
            
            # Auto-unload management
            self._access_count += 1
            if self._access_count % self._auto_unload_threshold == 0:
                self._manage_memory()
            
            return embeddings
    
    def unload_embeddings(self, key: str):
        """Manually unload specific embeddings from memory"""
        with self._lock:
            if key in self._embeddings:
                self._embeddings[key].unload()
    
    def unload_all(self):
        """Unload all embeddings from memory"""
        with self._lock:
            for compressed_emb in self._embeddings.values():
                compressed_emb.unload()
    
    def _manage_memory(self):
        """Internal memory management - unload least recently used"""
        # Simple strategy: unload all cached embeddings periodically
        for compressed_emb in self._embeddings.values():
            if compressed_emb._is_loaded:
                compressed_emb.unload()
    
    def save_to_disk(self, base_path: str):
        """Save all compressed embeddings to disk"""
        base = Path(base_path)
        base.mkdir(exist_ok=True)
        
        with self._lock:
            for key, compressed_emb in self._embeddings.items():
                filepath = base / f"{key}.emb"
                compressed_emb.save_to_file(str(filepath))
    
    def load_from_disk(self, base_path: str):
        """Load compressed embeddings from disk"""
        base = Path(base_path)
        
        with self._lock:
            for filepath in base.glob("*.emb"):
                key = filepath.stem
                compressed_emb = CompressedEmbeddings()
                compressed_emb.load_from_file(str(filepath))
                self._embeddings[key] = compressed_emb
                self._stats["total_embeddings"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        with self._lock:
            loaded_count = sum(1 for emb in self._embeddings.values() if emb._is_loaded)
            total_compressed_mb = sum(
                len(emb._compressed_data) / (1024 * 1024) 
                for emb in self._embeddings.values() 
                if emb._compressed_data
            )
            
            return {
                "total_embeddings": len(self._embeddings),
                "loaded_in_memory": loaded_count,
                "total_compressed_size_mb": total_compressed_mb,
                "access_count": self._access_count,
                **self._stats
            }

# Global instance for easy access
_global_embedding_manager: Optional[EmbeddingManager] = None

def get_embedding_manager() -> EmbeddingManager:
    """Get global embedding manager instance"""
    global _global_embedding_manager
    if _global_embedding_manager is None:
        _global_embedding_manager = EmbeddingManager()
    return _global_embedding_manager

def compress_embeddings(embeddings: np.ndarray, compression_level: int = 6) -> CompressedEmbeddings:
    """Utility function to compress embeddings"""
    return CompressedEmbeddings(embeddings, compression_level) 