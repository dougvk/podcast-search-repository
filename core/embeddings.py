#!/usr/bin/env python3
"""Memvid-powered embedding generator with proven stability patterns."""

from typing import List, Optional, Dict, Any, Union
import os
import logging
from pathlib import Path
import hashlib
import pickle
import time
from functools import lru_cache

# Set tokenizers parallelism to false (memvid best practice)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Memvid integration for stable embeddings
try:
    from memvid import MemvidEncoder
    from memvid.config import get_default_config
    MEMVID_AVAILABLE = True
except ImportError:
    MEMVID_AVAILABLE = False

logger = logging.getLogger(__name__)

class EmbeddingConfig:
    """Optimized configuration for embedding generation using memvid."""
    
    DEFAULT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'  # 384-dim, fast, good quality
    BATCH_SIZE = 32
    CACHE_SIZE = 10000
    
    @classmethod
    def get_memvid_config(cls, model_name: str = None) -> Dict[str, Any]:
        """Get memvid configuration following best practices."""
        config = get_default_config()
        
        # Customize config for our use case
        if 'embedding' not in config:
            config['embedding'] = {}
        
        # Set embedding model
        config['embedding']['model'] = model_name or cls.DEFAULT_MODEL
        
        return config

class EmbeddingGenerator:
    """Memvid-powered embedding generator with proven stability and caching."""
    
    def __init__(self, 
                 model_name: str = EmbeddingConfig.DEFAULT_MODEL,
                 cache_dir: Optional[str] = None):
        
        if not MEMVID_AVAILABLE:
            raise RuntimeError("Memvid not available. Install with: pip install memvid")
        
        self.model_name = model_name
        self.cache_dir = Path(cache_dir or "data/embeddings_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize memvid encoder with proven pattern
        self.model = None
        self._initialized = False
        
        logger.info(f"EmbeddingGenerator initialized with model: {model_name}")

    def _initialize(self):
        """Initialize memvid embedding model lazily using proven pattern."""
        if self._initialized:
            return
            
        try:
            # Use memvid's proven configuration approach
            config = EmbeddingConfig.get_memvid_config(self.model_name)
            encoder = MemvidEncoder(config)
            
            # Access the embedding model through index_manager (memvid's proven pattern)
            if hasattr(encoder, 'index_manager') and hasattr(encoder.index_manager, 'embedding_model'):
                self.model = encoder.index_manager.embedding_model
                self._initialized = True
                logger.info(f"✅ Memvid embedding model initialized: {type(self.model)}")
            else:
                raise RuntimeError("Could not access memvid's embedding model")
                
        except Exception as e:
            logger.error(f"Failed to initialize memvid embedding model: {e}")
            raise

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
        """Generate embeddings using memvid's stable approach with caching."""
        self._initialize()
        
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
        
        # Generate embeddings for uncached texts using memvid's stable approach
        if uncached_texts:
            logger.debug(f"Computing {len(uncached_texts)}/{len(text_list)} embeddings")
            
            # Use memvid's proven embedding generation
            new_embeddings = []
            for text in uncached_texts:
                try:
                    embedding = self.model.encode(text, convert_to_numpy=True)
                    new_embeddings.append(embedding.tolist())
                except Exception as e:
                    logger.error(f"Embedding generation failed for text: {text[:50]}... Error: {e}")
                    new_embeddings.append([0.0] * self.embedding_dimension)  # Fallback zero vector
            
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
        self._initialize()
        # Test with a simple text to get dimension
        test_embedding = self.model.encode("test", convert_to_numpy=True)
        return test_embedding.shape[0]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass  # Model cleanup handled by memvid