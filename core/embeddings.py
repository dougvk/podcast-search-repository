#!/usr/bin/env python3
"""Minimal, efficient embedding generator with sentence-transformers."""

from typing import List, Optional, Dict, Any, Union
import torch
import logging
from sentence_transformers import SentenceTransformer
from pathlib import Path
import hashlib
import pickle
import time
from functools import lru_cache

logger = logging.getLogger(__name__)

class EmbeddingConfig:
    """Optimized configuration for embedding generation."""
    
    DEFAULT_MODEL = 'all-MiniLM-L6-v2'  # 384-dim, fast, good quality
    BATCH_SIZE = 32
    MAX_SEQ_LENGTH = 256
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CACHE_SIZE = 10000
    
    @classmethod
    def get_device_config(cls) -> Dict[str, Any]:
        """Get optimized device configuration."""
        return {
            'device': cls.DEVICE,
            'show_progress_bar': False,  # Performance
            'convert_to_numpy': True,    # Memory efficiency
            'normalize_embeddings': True  # Better similarity search
        }

class EmbeddingGenerator:
    """Ultra-efficient embedding generator with caching and batch processing."""
    
    def __init__(self, 
                 model_name: str = EmbeddingConfig.DEFAULT_MODEL,
                 cache_dir: Optional[str] = None,
                 device: Optional[str] = None):
        
        self.model_name = model_name
        self.device = device or EmbeddingConfig.DEVICE
        self.cache_dir = Path(cache_dir or "data/embeddings_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model with optimized settings
        logger.info(f"Loading {model_name} on {self.device}")
        self.model = SentenceTransformer(
            model_name, 
            device=self.device,
            trust_remote_code=True
        )
        
        # Configure for performance
        self.model.max_seq_length = EmbeddingConfig.MAX_SEQ_LENGTH
        self._memory_cache = {}  # In-memory LRU cache
        
        logger.info(f"✅ Model loaded: {self.model.get_sentence_embedding_dimension()}D vectors")

    @lru_cache(maxsize=EmbeddingConfig.CACHE_SIZE)
    def _get_cached_embedding(self, text_hash: str) -> Optional[List[float]]:
        """Memory-efficient LRU cache for embeddings."""
        cache_file = self.cache_dir / f"{text_hash}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                cache_file.unlink(missing_ok=True)
        return None

    def _save_embedding(self, text_hash: str, embedding: List[float]):
        """Save embedding to disk cache."""
        try:
            cache_file = self.cache_dir / f"{text_hash}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.error(f"Cache save failed: {e}")

    def _get_text_hash(self, text: str) -> str:
        """Fast text hashing for cache keys."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def generate_embeddings(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings with automatic caching and batching."""
        single_input = isinstance(texts, str)
        text_list = [texts] if single_input else texts
        
        if not text_list:
            return [] if not single_input else []
        
        # Check cache first
        results = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(text_list):
            text_hash = self._get_text_hash(text)
            cached = self._get_cached_embedding(text_hash)
            
            if cached is not None:
                results.append(cached)
            else:
                results.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            logger.debug(f"Computing {len(uncached_texts)}/{len(text_list)} embeddings")
            
            # Batch encode with optimized settings
            new_embeddings = self.model.encode(
                uncached_texts,
                batch_size=min(EmbeddingConfig.BATCH_SIZE, len(uncached_texts)),
                **EmbeddingConfig.get_device_config()
            ).tolist()
            
            # Cache and insert results
            for i, (text, embedding) in enumerate(zip(uncached_texts, new_embeddings)):
                text_hash = self._get_text_hash(text)
                self._save_embedding(text_hash, embedding)
                results[uncached_indices[i]] = embedding
        
        return results[0] if single_input else results

    def batch_process_chunks(self, chunks: List[Dict[str, Any]], 
                           text_field: str = 'text') -> List[Dict[str, Any]]:
        """Memory-efficient batch processing with progress tracking."""
        if not chunks:
            return []
        
        logger.info(f"Processing {len(chunks)} chunks...")
        start_time = time.time()
        
        # Extract texts for batching
        texts = [chunk[text_field] for chunk in chunks]
        embeddings = self.generate_embeddings(texts)
        
        # Add embeddings to chunks
        enriched_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            enriched_chunk = chunk.copy()
            enriched_chunk['embedding'] = embedding
            enriched_chunks.append(enriched_chunk)
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Processed {len(chunks)} chunks in {processing_time:.2f}s")
        
        return enriched_chunks

    def get_similarity(self, text1: str, text2: str) -> float:
        """Fast cosine similarity between two texts."""
        emb1, emb2 = self.generate_embeddings([text1, text2])
        
        # Cosine similarity (normalized vectors)
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        return max(0.0, min(1.0, dot_product))  # Clamp to [0,1]

    def clear_cache(self):
        """Clear all cached embeddings."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        self._get_cached_embedding.cache_clear()
        logger.info("Cache cleared")

    @property 
    def embedding_dimension(self) -> int:
        """Get embedding vector dimension."""
        return self.model.get_sentence_embedding_dimension()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass  # Model cleanup handled by torch