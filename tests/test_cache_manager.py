#!/usr/bin/env python3
"""
Tests for high-performance cache manager
"""

import unittest
import tempfile
import time
import threading
from unittest.mock import Mock, patch, MagicMock

from core.cache_manager import (
    CacheConfig, LRUCache, CacheManager, 
    get_cache_manager, init_cache
)

class TestLRUCache(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.cache = LRUCache(max_size=3, ttl=1, compress_threshold=10)
    
    def test_basic_operations(self):
        """Test basic cache operations"""
        # Test set and get
        self.cache.set("key1", "value1")
        self.assertEqual(self.cache.get("key1"), "value1")
        
        # Test non-existent key
        self.assertIsNone(self.cache.get("nonexistent"))
    
    def test_lru_eviction(self):
        """Test LRU eviction policy"""
        # Fill cache to capacity
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2") 
        self.cache.set("key3", "value3")
        
        # Access key1 to make it most recent
        self.cache.get("key1")
        
        # Add new item - should evict key2 (least recent)
        self.cache.set("key4", "value4")
        
        self.assertIsNotNone(self.cache.get("key1"))
        self.assertIsNone(self.cache.get("key2"))  # Evicted
        self.assertIsNotNone(self.cache.get("key3"))
        self.assertIsNotNone(self.cache.get("key4"))
    
    def test_ttl_expiration(self):
        """Test TTL-based expiration"""
        self.cache.set("key1", "value1")
        self.assertEqual(self.cache.get("key1"), "value1")
        
        # Wait for expiration
        time.sleep(1.1)
        self.assertIsNone(self.cache.get("key1"))
    
    def test_compression(self):
        """Test data compression for large values"""
        large_data = "x" * 100  # Above compress_threshold
        self.cache.set("large", large_data)
        self.assertEqual(self.cache.get("large"), large_data)
    
    def test_thread_safety(self):
        """Test thread safety"""
        results = {}
        
        def worker(thread_id):
            for i in range(10):
                key = f"thread_{thread_id}_key_{i}"
                value = f"thread_{thread_id}_value_{i}"
                self.cache.set(key, value)
                results[key] = self.cache.get(key)
        
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Verify no corruption occurred
        for key, value in results.items():
            if value is not None:  # Some may be evicted due to size limit
                self.assertIn(key.split("_key_")[0], value)
    
    def test_stats(self):
        """Test cache statistics"""
        stats = self.cache.stats()
        self.assertEqual(stats['size'], 0)
        self.assertEqual(stats['active'], 0)
        
        self.cache.set("key1", "value1")
        stats = self.cache.stats()
        self.assertEqual(stats['size'], 1)
        self.assertEqual(stats['active'], 1)

class TestCacheManager(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.config = CacheConfig(memory_size=5, ttl=10)
        self.cache_manager = CacheManager(self.config)
    
    def test_memory_cache_operations(self):
        """Test memory cache functionality"""
        # Test search result caching
        results = [{"text": "test", "score": 0.9}]
        self.cache_manager.cache_search_results("test query", results)
        
        cached = self.cache_manager.get_search_results("test query")
        self.assertEqual(cached, results)
        
        # Test embedding caching
        embeddings = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.cache_manager.cache_embeddings("test text", embeddings)
        
        cached_emb = self.cache_manager.get_embeddings("test text")
        self.assertEqual(cached_emb, embeddings)
    
    def test_query_normalization(self):
        """Test query normalization for cache keys"""
        results = [{"text": "test", "score": 0.9}]
        
        # Cache with uppercase and spaces
        self.cache_manager.cache_search_results("  TEST QUERY  ", results)
        
        # Should find with different casing/spacing
        cached = self.cache_manager.get_search_results("test query")
        self.assertEqual(cached, results)
    
    @patch('redis.from_url')
    def test_redis_fallback(self, mock_redis):
        """Test Redis connection with fallback"""
        # Mock Redis connection failure
        mock_redis.side_effect = Exception("Redis unavailable")
        
        config = CacheConfig(redis_url="redis://localhost:6379")
        cache_manager = CacheManager(config)
        
        # Should fallback to memory cache
        self.assertIsNone(cache_manager.redis)
        
        # Operations should still work
        cache_manager.cache_search_results("test", [{"text": "test"}])
        cached = cache_manager.get_search_results("test")
        self.assertIsNotNone(cached)
    
    @patch('redis.from_url')
    def test_redis_operations(self, mock_redis):
        """Test Redis operations when available"""
        # Mock successful Redis connection
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.get.return_value = None
        mock_client.setex.return_value = True
        mock_redis.return_value = mock_client
        
        config = CacheConfig(redis_url="redis://localhost:6379")
        cache_manager = CacheManager(config)
        
        # Verify Redis was initialized
        self.assertIsNotNone(cache_manager.redis)
        mock_client.ping.assert_called_once()
        
        # Test set operation
        cache_manager.set("test", "value", "key")
        mock_client.setex.assert_called()
    
    def test_stats(self):
        """Test comprehensive statistics"""
        stats = self.cache_manager.stats()
        
        # Check structure
        self.assertIn('memory', stats)
        self.assertIn('redis', stats)
        self.assertIn('config', stats)
        
        # Check memory stats
        self.assertIn('size', stats['memory'])
        self.assertIn('active', stats['memory'])
        
        # Check config
        self.assertEqual(stats['config']['memory_size'], 5)
        self.assertEqual(stats['config']['ttl'], 10)
    
    def test_clear_cache(self):
        """Test cache clearing"""
        # Add some data
        self.cache_manager.cache_search_results("test", [{"text": "test"}])
        self.assertIsNotNone(self.cache_manager.get_search_results("test"))
        
        # Clear cache
        self.cache_manager.clear()
        self.assertIsNone(self.cache_manager.get_search_results("test"))

class TestCacheManagerGlobal(unittest.TestCase):
    def setUp(self):
        """Reset global cache state"""
        import core.cache_manager
        core.cache_manager._cache_manager = None
    
    def test_global_cache_manager(self):
        """Test global cache manager singleton"""
        # First call creates instance
        cache1 = get_cache_manager()
        self.assertIsNotNone(cache1)
        
        # Second call returns same instance
        cache2 = get_cache_manager()
        self.assertIs(cache1, cache2)
    
    def test_init_cache(self):
        """Test cache initialization with config"""
        cache = init_cache(memory_size=100, ttl=300)
        self.assertIsNotNone(cache)
        self.assertEqual(cache.config.memory_size, 100)
        self.assertEqual(cache.config.ttl, 300)

class TestCachePerformance(unittest.TestCase):
    def setUp(self):
        """Set up performance test environment"""
        self.cache = LRUCache(max_size=1000, ttl=3600)
    
    def test_large_dataset_performance(self):
        """Test performance with large dataset"""
        start_time = time.time()
        
        # Insert 500 items
        for i in range(500):
            self.cache.set(f"key_{i}", f"value_{i}" * 10)
        
        insert_time = time.time() - start_time
        
        # Retrieve all items
        start_time = time.time()
        for i in range(500):
            value = self.cache.get(f"key_{i}")
            self.assertIsNotNone(value)
        
        retrieve_time = time.time() - start_time
        
        # Performance assertions (should be fast)
        self.assertLess(insert_time, 1.0, "Insert time should be under 1 second")
        self.assertLess(retrieve_time, 0.5, "Retrieve time should be under 0.5 seconds")
    
    def test_concurrent_access_performance(self):
        """Test performance under concurrent access"""
        def worker():
            for i in range(50):
                key = f"thread_key_{i}"
                self.cache.set(key, f"value_{i}")
                self.cache.get(key)
        
        start_time = time.time()
        
        # Run 10 concurrent workers
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        total_time = time.time() - start_time
        
        # Should handle concurrent access efficiently
        self.assertLess(total_time, 2.0, "Concurrent access should complete under 2 seconds")

if __name__ == '__main__':
    unittest.main() 