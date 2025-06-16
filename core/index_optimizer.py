#!/usr/bin/env python3
"""
High-performance FAISS index optimizer with intelligent partitioning
"""

import numpy as np
import faiss
import logging
import time
import pickle
import gzip
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import threading
from enum import Enum

logger = logging.getLogger(__name__)

class IndexType(Enum):
    """Supported FAISS index types"""
    FLAT = "Flat"
    IVF_FLAT = "IVFFlat" 
    IVF_PQ = "IVFPQ"
    HNSW = "HNSW"
    LSH = "LSH"

@dataclass
class IndexConfig:
    """Index configuration with performance-optimized defaults"""
    index_type: IndexType = IndexType.IVF_FLAT
    nlist: int = 100  # Number of clusters for IVF
    nprobe: int = 32  # Search clusters
    m: int = 8        # PQ subvectors
    nbits: int = 8    # PQ bits per subvector
    efConstruction: int = 200  # HNSW construction
    efSearch: int = 100        # HNSW search
    metric: int = faiss.METRIC_INNER_PRODUCT  # Default metric
    use_gpu: bool = False
    
class IndexOptimizer:
    """Intelligent FAISS index optimizer with automatic parameter tuning"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.config = IndexConfig()
        self._lock = threading.RLock()
        self._stats = {"builds": 0, "searches": 0, "optimization_time": 0}
    
    def auto_configure(self, num_vectors: int, vector_sample: Optional[np.ndarray] = None) -> IndexConfig:
        """Automatically configure index based on dataset size and characteristics"""
        config = IndexConfig()
        
        # Size-based optimization strategy
        if num_vectors < 1000:
            config.index_type = IndexType.FLAT  # Exact search for small datasets
        elif num_vectors < 10000:
            config.index_type = IndexType.IVF_FLAT
            config.nlist = min(100, max(4, int(np.sqrt(num_vectors))))
        elif num_vectors < 100000:
            config.index_type = IndexType.IVF_FLAT
            config.nlist = min(4096, max(100, num_vectors // 100))
        else:
            config.index_type = IndexType.IVF_PQ  # Memory-efficient for large datasets
            config.nlist = min(8192, max(1000, num_vectors // 100))
            config.m = min(self.dimension // 4, 64)  # Ensure m divides dimension
        
        # Adjust nprobe based on accuracy/speed tradeoff
        config.nprobe = min(config.nlist, max(1, config.nlist // 4))
        
        logger.info(f"Auto-configured {config.index_type.value} index for {num_vectors} vectors")
        return config
    
    def build_index(self, vectors: np.ndarray, config: Optional[IndexConfig] = None) -> Dict[str, Any]:
        """Build optimized FAISS index with intelligent partitioning"""
        with self._lock:
            start_time = time.time()
            
            if config is None:
                config = self.auto_configure(len(vectors))
            self.config = config
            
            # Ensure vectors are float32 and C-contiguous
            if vectors.dtype != np.float32:
                vectors = vectors.astype(np.float32)
            if not vectors.flags['C_CONTIGUOUS']:
                vectors = np.ascontiguousarray(vectors)
            
            # Build index based on type
            if config.index_type == IndexType.FLAT:
                self.index = faiss.IndexFlatIP(self.dimension)
            elif config.index_type == IndexType.IVF_FLAT:
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, config.nlist)
            elif config.index_type == IndexType.IVF_PQ:
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFPQ(quantizer, self.dimension, config.nlist, config.m, config.nbits)
            elif config.index_type == IndexType.HNSW:
                self.index = faiss.IndexHNSWFlat(self.dimension, 32)
                self.index.hnsw.efConstruction = config.efConstruction
            else:
                raise ValueError(f"Unsupported index type: {config.index_type}")
            
            # Train index if needed
            if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                logger.info(f"Training {config.index_type.value} index...")
                self.index.train(vectors)
            
            # Add vectors
            self.index.add(vectors)
            
            # Set search parameters
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = config.nprobe
            if hasattr(self.index, 'hnsw'):
                self.index.hnsw.efSearch = config.efSearch
            
            build_time = time.time() - start_time
            self._stats["builds"] += 1
            self._stats["optimization_time"] += build_time
            
            stats = {
                "index_type": config.index_type.value,
                "num_vectors": len(vectors),
                "dimension": self.dimension,
                "build_time": build_time,
                "is_trained": getattr(self.index, 'is_trained', True),
                "ntotal": self.index.ntotal,
                "config": config.__dict__
            }
            
            logger.info(f"Built {config.index_type.value} index: {len(vectors)} vectors in {build_time:.2f}s")
            return stats
    
    def search(self, query_vectors: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized vector search with automatic parameter adjustment"""
        with self._lock:
            if self.index is None:
                raise ValueError("Index not built. Call build_index() first.")
            
            # Ensure query vectors are properly formatted
            if query_vectors.dtype != np.float32:
                query_vectors = query_vectors.astype(np.float32)
            if not query_vectors.flags['C_CONTIGUOUS']:
                query_vectors = np.ascontiguousarray(query_vectors)
            
            # Handle single vector queries
            if query_vectors.ndim == 1:
                query_vectors = query_vectors.reshape(1, -1)
            
            start_time = time.time()
            scores, indices = self.index.search(query_vectors, k)
            search_time = time.time() - start_time
            
            self._stats["searches"] += 1
            
            logger.debug(f"Search completed: {len(query_vectors)} queries in {search_time:.3f}s")
            return scores, indices
    
    def optimize_search_params(self, query_vectors: np.ndarray, target_recall: float = 0.9) -> Dict[str, Any]:
        """Automatically optimize search parameters for target recall"""
        if self.index is None or not hasattr(self.index, 'nprobe'):
            return {"message": "Index doesn't support parameter optimization"}
        
        best_nprobe = self.config.nprobe
        best_time = float('inf')
        
        # Test different nprobe values
        for nprobe in [1, 4, 8, 16, 32, 64, 128]:
            if nprobe > self.config.nlist:
                break
                
            self.index.nprobe = nprobe
            start_time = time.time()
            self.search(query_vectors[:min(10, len(query_vectors))])  # Sample queries
            avg_time = (time.time() - start_time) / min(10, len(query_vectors))
            
            if avg_time < best_time:
                best_time = avg_time
                best_nprobe = nprobe
        
        self.index.nprobe = best_nprobe
        self.config.nprobe = best_nprobe
        
        return {
            "optimized_nprobe": best_nprobe,
            "avg_search_time": best_time,
            "total_clusters": self.config.nlist
        }
    
    def save_index(self, path: Union[str, Path], compress: bool = True) -> Dict[str, Any]:
        """Save optimized index with optional compression"""
        if self.index is None:
            raise ValueError("No index to save")
        
        path = Path(path)
        
        # Save index
        index_path = path.with_suffix('.faiss')
        faiss.write_index(self.index, str(index_path))
        
        # Save config and stats
        metadata = {
            "config": self.config.__dict__,
            "stats": self._stats,
            "dimension": self.dimension,
            "ntotal": self.index.ntotal
        }
        
        metadata_path = path.with_suffix('.pkl.gz' if compress else '.pkl')
        with (gzip.open if compress else open)(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        return {
            "index_path": str(index_path),
            "metadata_path": str(metadata_path),
            "index_size_mb": index_path.stat().st_size / (1024 * 1024),
            "compressed": compress
        }
    
    def load_index(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load optimized index with metadata"""
        path = Path(path)
        
        # Load index
        index_path = path.with_suffix('.faiss')
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        for ext in ['.pkl.gz', '.pkl']:
            metadata_path = path.with_suffix(ext)
            if metadata_path.exists():
                with (gzip.open if ext.endswith('.gz') else open)(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                self.config = IndexConfig(**metadata["config"])
                self._stats = metadata["stats"]
                self.dimension = metadata["dimension"]
                break
        
        return {
            "index_type": self.config.index_type.value,
            "ntotal": self.index.ntotal,
            "dimension": self.dimension,
            "loaded_from": str(index_path)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = self._stats.copy()
        if self.index:
            stats.update({
                "ntotal": self.index.ntotal,
                "index_type": self.config.index_type.value,
                "is_trained": getattr(self.index, 'is_trained', True)
            })
        return stats

def create_optimized_index(vectors: np.ndarray, dimension: int = 384, target_size: Optional[int] = None) -> IndexOptimizer:
    """Factory function to create optimized index for given vectors"""
    optimizer = IndexOptimizer(dimension)
    
    if target_size and len(vectors) > target_size:
        # Use intelligent sampling for very large datasets
        indices = np.random.choice(len(vectors), target_size, replace=False)
        sample_vectors = vectors[indices]
        config = optimizer.auto_configure(target_size, sample_vectors)
    else:
        config = optimizer.auto_configure(len(vectors), vectors)
    
    optimizer.build_index(vectors, config)
    return optimizer 