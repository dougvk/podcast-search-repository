#!/usr/bin/env python3
"""
High-performance caching manager for search results and embeddings
"""

import json
import pickle
import gzip
import hashlib
import time
import threading
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Cache configuration with performance-optimized defaults"""
    redis_url: Optional[str] = None
    memory_size: int = 1000  # Max items in memory cache
    ttl: int = 3600  # 1 hour default TTL
    compress: bool = True  # Auto-compress large values
    compress_threshold: int = 1024  # Compress if >1KB
    
class LRUCache:
    """Thread-safe LRU cache with compression"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600, compress_threshold: int = 1024):
        self.max_size = max_size
        self.ttl = ttl
        self.compress_threshold = compress_threshold
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        
    def _compress(self, data: bytes) -> bytes:
        """Compress data if above threshold"""
        return gzip.compress(data) if len(data) > self.compress_threshold else data
        
    def _decompress(self, data: bytes) -> bytes:
        """Decompress data if compressed"""
        try:
            return gzip.decompress(data)
        except gzip.BadGzipFile:
            return data
            
    def _is_expired(self, timestamp: float) -> bool:
        """Check if timestamp is expired"""
        return time.time() - timestamp > self.ttl
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with LRU update"""
        with self._lock:
            if key not in self._cache:
                return None
                
            value, timestamp = self._cache[key]
            if self._is_expired(timestamp):
                del self._cache[key]
                return None
                
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return pickle.loads(self._decompress(value))
            
    def set(self, key: str, value: Any) -> None:
        """Set value in cache with LRU eviction"""
        with self._lock:
            data = self._compress(pickle.dumps(value))
            self._cache[key] = (data, time.time())
            self._cache.move_to_end(key)
            
            # Evict oldest if over size limit
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)
                
    def clear(self) -> None:
        """Clear all cached items"""
        with self._lock:
            self._cache.clear()
            
    def stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        with self._lock:
            expired = sum(1 for _, (_, ts) in self._cache.items() if self._is_expired(ts))
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'expired': expired,
                'active': len(self._cache) - expired
            }

class CacheManager:
    """High-performance cache manager with Redis + in-memory fallback"""
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.memory_cache = LRUCache(
            self.config.memory_size,
            self.config.ttl,
            self.config.compress_threshold
        )
        self.redis = self._init_redis()
        
    def _init_redis(self):
        """Initialize Redis connection with fallback"""
        if not self.config.redis_url:
            return None
            
        try:
            import redis
            client = redis.from_url(self.config.redis_url, decode_responses=False)
            client.ping()
            logger.info("Redis cache connected")
            return client
        except Exception as e:
            logger.warning(f"Redis unavailable, using memory cache: {e}")
            return None
            
    def _make_key(self, prefix: str, *args) -> str:
        """Create cache key from prefix and arguments"""
        key_data = f"{prefix}:{':'.join(str(arg) for arg in args)}"
        return hashlib.md5(key_data.encode()).hexdigest()
        
    def get(self, prefix: str, *args) -> Optional[Any]:
        """Get from cache (Redis first, then memory)"""
        key = self._make_key(prefix, *args)
        
        # Try Redis first
        if self.redis:
            try:
                data = self.redis.get(key)
                if data:
                    return pickle.loads(gzip.decompress(data) if data.startswith(b'\x1f\x8b') else data)
            except Exception as e:
                logger.warning(f"Redis get error: {e}")
                
        # Fallback to memory cache
        return self.memory_cache.get(key)
        
    def set(self, prefix: str, value: Any, *args, ttl: Optional[int] = None) -> None:
        """Set in cache (both Redis and memory)"""
        key = self._make_key(prefix, *args)
        ttl = ttl or self.config.ttl
        
        # Set in memory cache
        self.memory_cache.set(key, value)
        
        # Set in Redis if available
        if self.redis:
            try:
                data = pickle.dumps(value)
                if self.config.compress and len(data) > self.config.compress_threshold:
                    data = gzip.compress(data)
                self.redis.setex(key, ttl, data)
            except Exception as e:
                logger.warning(f"Redis set error: {e}")
                
    def cache_search_results(self, query: str, results: List[Dict], max_results: int = 10, threshold: float = 0.7) -> None:
        """Cache search results with query normalization"""
        normalized_query = query.lower().strip()
        self.set("search", results, normalized_query, max_results, threshold)
        
    def get_search_results(self, query: str, max_results: int = 10, threshold: float = 0.7) -> Optional[List[Dict]]:
        """Get cached search results"""
        normalized_query = query.lower().strip()
        return self.get("search", normalized_query, max_results, threshold)
        
    def cache_embeddings(self, text: str, embeddings: Union[List[float], Any]) -> None:
        """Cache text embeddings"""
        self.set("embeddings", embeddings, text)
        
    def get_embeddings(self, text: str) -> Optional[Union[List[float], Any]]:
        """Get cached embeddings"""
        return self.get("embeddings", text)
        
    def clear(self, prefix: Optional[str] = None) -> None:
        """Clear cache (specific prefix or all)"""
        if prefix is None:
            self.memory_cache.clear()
            if self.redis:
                try:
                    self.redis.flushdb()
                except Exception as e:
                    logger.warning(f"Redis clear error: {e}")
        else:
            # Clear by prefix is complex, not implemented for simplicity
            logger.warning("Prefix-specific clearing not implemented")
            
    def stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        memory_stats = self.memory_cache.stats()
        redis_stats = {}
        
        if self.redis:
            try:
                info = self.redis.info()
                redis_stats = {
                    'connected': True,
                    'used_memory': info.get('used_memory', 0),
                    'keys': self.redis.dbsize(),
                    'hits': info.get('keyspace_hits', 0),
                    'misses': info.get('keyspace_misses', 0)
                }
            except Exception as e:
                redis_stats = {'connected': False, 'error': str(e)}
        else:
            redis_stats = {'connected': False}
            
        return {
            'memory': memory_stats,
            'redis': redis_stats,
            'config': {
                'memory_size': self.config.memory_size,
                'ttl': self.config.ttl,
                'compress': self.config.compress
            }
        }

# Global cache instance
_cache_manager = None

def get_cache_manager(config: CacheConfig = None) -> CacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(config)
    return _cache_manager

def init_cache(redis_url: Optional[str] = None, memory_size: int = 1000, ttl: int = 3600) -> CacheManager:
    """Initialize cache with configuration"""
    config = CacheConfig(redis_url=redis_url, memory_size=memory_size, ttl=ttl)
    global _cache_manager
    _cache_manager = CacheManager(config)
    return _cache_manager 