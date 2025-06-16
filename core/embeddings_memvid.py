#!/usr/bin/env python3
"""Memvid-inspired embedding generator with proven stability patterns."""

import os
import logging
from typing import List, Union, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
from functools import lru_cache
import pickle
from pathlib import Path

# Set tokenizers parallelism to false (memvid best practice)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation following memvid patterns."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "auto"  # auto, cpu, cuda
    cache_dir: str = "data/embedding_cache"
    batch_size: int = 32
    max_seq_length: int = 512
    normalize_embeddings: bool = True
    use_memvid_fallback: bool = True

class MemvidStyleEmbeddingGenerator:
    """Embedding generator following memvid's proven patterns for stability."""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.model = None
        self._cache_dir = Path(self.config.cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model lazily (memvid pattern)
        self._model_initialized = False
        
        logger.info(f"EmbeddingGenerator initialized with model: {self.config.model_name}")

    def _initialize_model(self):
        """Lazy model initialization following memvid's approach."""
        if self._model_initialized:
            return
        
        try:
            # Try memvid's approach first
            if self.config.use_memvid_fallback:
                success = self._try_memvid_initialization()
                if success:
                    return
            
            # Fallback to direct sentence-transformers
            self._initialize_sentence_transformers()
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise RuntimeError(f"Could not initialize embedding model: {e}")

    def _try_memvid_initialization(self) -> bool:
        """Try to use memvid's proven embedding approach."""
        try:
            from memvid import MemvidEncoder
            from memvid.config import get_default_config
            
            logger.info("🔄 Attempting memvid-based initialization...")
            
            # Use memvid's configuration
            memvid_config = get_default_config()
            
            # Create a memvid encoder to access its embedding model
            encoder = MemvidEncoder(memvid_config)
            
            # Access the embedding model that memvid successfully initialized
            if hasattr(encoder, 'embedding_model'):
                self.model = encoder.embedding_model
                self._model_initialized = True
                logger.info("✅ Successfully initialized using memvid's embedding model")
                return True
            
        except ImportError:
            logger.info("⚠️  Memvid not available, falling back to direct initialization")
        except Exception as e:
            logger.warning(f"⚠️  Memvid initialization failed: {e}")
        
        return False

    def _initialize_sentence_transformers(self):
        """Direct sentence-transformers initialization with safety measures."""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"🔄 Initializing sentence-transformers model: {self.config.model_name}")
            
            # Use memvid's device detection pattern
            device = self._detect_device()
            
            # Initialize with safety parameters
            self.model = SentenceTransformer(
                self.config.model_name,
                device=device,
                cache_folder=str(self._cache_dir / "models")
            )
            
            # Configure model settings
            if hasattr(self.model, 'max_seq_length'):
                self.model.max_seq_length = self.config.max_seq_length
            
            self._model_initialized = True
            logger.info(f"✅ Model initialized on device: {device}")
            
        except Exception as e:
            logger.error(f"Direct initialization failed: {e}")
            raise

    def _detect_device(self) -> str:
        """Detect optimal device following memvid's approach."""
        if self.config.device != "auto":
            return self.config.device
        
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("🔥 CUDA available, using GPU")
                return "cuda"
        except ImportError:
            pass
        
        logger.info("💻 Using CPU device")
        return "cpu"

    @lru_cache(maxsize=1000)
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[List[float]]:
        """Load embedding from cache."""
        cache_file = self._cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")
        return None

    def _save_to_cache(self, cache_key: str, embedding: List[float]):
        """Save embedding to cache."""
        cache_file = self._cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")

    def generate_embeddings(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings with memvid-style error handling."""
        # Ensure model is initialized
        self._initialize_model()
        
        # Handle single text input
        if isinstance(texts, str):
            return self._generate_single_embedding(texts)
        
        # Handle batch input
        return self._generate_batch_embeddings(texts)

    def _generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for single text with caching."""
        if not text.strip():
            return [0.0] * 384  # Return zero vector for empty text
        
        # Check cache first
        cache_key = self._get_cache_key(text)
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return cached
        
        try:
            # Generate embedding
            embedding = self.model.encode(
                text,
                normalize_embeddings=self.config.normalize_embeddings,
                convert_to_numpy=True
            )
            
            # Convert to list and cache
            embedding_list = embedding.tolist()
            self._save_to_cache(cache_key, embedding_list)
            
            return embedding_list
            
        except Exception as e:
            logger.error(f"Embedding generation failed for text: {text[:50]}... Error: {e}")
            # Return zero vector as fallback
            return [0.0] * 384

    def _generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch of texts."""
        if not texts:
            return []
        
        embeddings = []
        
        # Process in batches to avoid memory issues
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            
            try:
                # Generate batch embeddings
                batch_embeddings = self.model.encode(
                    batch,
                    normalize_embeddings=self.config.normalize_embeddings,
                    convert_to_numpy=True,
                    batch_size=len(batch)
                )
                
                # Convert to lists and add to results
                for j, embedding in enumerate(batch_embeddings):
                    embedding_list = embedding.tolist()
                    embeddings.append(embedding_list)
                    
                    # Cache individual embeddings
                    cache_key = self._get_cache_key(batch[j])
                    self._save_to_cache(cache_key, embedding_list)
                
            except Exception as e:
                logger.error(f"Batch embedding failed for batch {i//self.config.batch_size}: {e}")
                # Add zero vectors for failed batch
                for _ in batch:
                    embeddings.append([0.0] * 384)
        
        return embeddings

    def __enter__(self):
        """Context manager entry - initialize model."""
        self._initialize_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        # Cleanup model if needed
        if hasattr(self.model, 'cpu'):
            try:
                self.model.cpu()  # Move to CPU to free GPU memory
            except:
                pass
        
        # Clear model reference
        self.model = None
        self._model_initialized = False
        
        logger.info("EmbeddingGenerator resources cleaned up")

    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics."""
        cache_files = list(self._cache_dir.glob("*.pkl"))
        
        return {
            'model_name': self.config.model_name,
            'model_initialized': self._model_initialized,
            'device': self.config.device,
            'cache_size': len(cache_files),
            'cache_dir': str(self._cache_dir),
            'batch_size': self.config.batch_size,
            'normalize_embeddings': self.config.normalize_embeddings
        }

    def clear_cache(self):
        """Clear embedding cache."""
        try:
            import shutil
            if self._cache_dir.exists():
                shutil.rmtree(self._cache_dir)
                self._cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Embedding cache cleared")
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")

# Convenience function following memvid's pattern
def create_embedding_generator(model_name: str = None, use_memvid: bool = True) -> MemvidStyleEmbeddingGenerator:
    """Create embedding generator with memvid-style configuration."""
    config = EmbeddingConfig()
    
    if model_name:
        config.model_name = model_name
    
    config.use_memvid_fallback = use_memvid
    
    return MemvidStyleEmbeddingGenerator(config) 