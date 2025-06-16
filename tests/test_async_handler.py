#!/usr/bin/env python3
"""
Comprehensive tests for async request handling and concurrent processing
"""

import unittest
import asyncio
import time
import threading
import tempfile
import os
import shutil
from unittest.mock import Mock, AsyncMock
from typing import List, Dict

from core.async_handler import (
    AsyncRequestHandler, ConcurrentSearchProcessor,
    get_async_handler, concurrent_search,
    async_cached, rate_limit
)

class TestAsyncRequestHandler(unittest.TestCase):
    """Test async request handler functionality"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_files = []
        self.handler = AsyncRequestHandler(max_workers=5, queue_size=10)
    
    def tearDown(self):
        # Clean up temporary files
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        # Shutdown handler
        self.handler.shutdown()
    
    def test_sync_function_handling(self):
        """Test handling synchronous functions"""
        def sync_func(x, y):
            return x + y
        
        async def test():
            result = await self.handler.handle_request(sync_func, 5, 10)
            self.assertEqual(result, 15)
        
        asyncio.run(test())
    
    def test_async_function_handling(self):
        """Test handling asynchronous functions"""
        async def async_func(x, y):
            await asyncio.sleep(0.01)
            return x * y
        
        async def test():
            result = await self.handler.handle_request(async_func, 3, 4)
            self.assertEqual(result, 12)
        
        asyncio.run(test())
    
    def test_batch_processing(self):
        """Test concurrent batch request processing"""
        def compute(x):
            time.sleep(0.01)  # Simulate work
            return x * 2
        
        async def test():
            requests = [(compute, (i,), {}) for i in range(5)]
            results = await self.handler.batch_requests(requests)
            
            self.assertEqual(len(results), 5)
            for i, result in enumerate(results):
                self.assertEqual(result, i * 2)
        
        asyncio.run(test())
    
    def test_statistics_tracking(self):
        """Test request statistics tracking"""
        def dummy_func():
            return "result"
        
        async def test():
            # Process some requests
            for _ in range(3):
                await self.handler.handle_request(dummy_func)
            
            stats = self.handler.get_stats()
            self.assertEqual(stats["total_requests"], 3)
            self.assertGreater(stats["avg_response_time"], 0)
            self.assertGreaterEqual(stats["concurrent_peak"], 0)
        
        asyncio.run(test())
    
    def test_error_handling(self):
        """Test error handling in async requests"""
        def failing_func():
            raise ValueError("Test error")
        
        async def test():
            with self.assertRaises(ValueError):
                await self.handler.handle_request(failing_func)
        
        asyncio.run(test())
    
    def test_semaphore_limiting(self):
        """Test request queue limiting with semaphore"""
        handler = AsyncRequestHandler(max_workers=2, queue_size=3)
        
        def slow_func():
            time.sleep(0.1)
            return "done"
        
        async def test():
            start_time = time.time()
            
            # Start multiple requests (more than queue size)
            tasks = [handler.handle_request(slow_func) for _ in range(5)]
            results = await asyncio.gather(*tasks)
            
            elapsed = time.time() - start_time
            
            # Should complete all tasks
            self.assertEqual(len(results), 5)
            self.assertTrue(all(r == "done" for r in results))
            
            # Should take time due to limiting
            self.assertGreater(elapsed, 0.1)
        
        try:
            asyncio.run(test())
        finally:
            handler.shutdown()

class TestConcurrentSearchProcessor(unittest.TestCase):
    """Test concurrent search processing functionality"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_files = []
        
        # Mock search function
        def mock_search(query: str, **kwargs):
            # Simulate search results
            return [(f"result for {query}", 0.8), (f"another result for {query}", 0.6)]
        
        self.processor = ConcurrentSearchProcessor(mock_search, max_concurrent=5)
    
    def tearDown(self):
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_concurrent_search(self):
        """Test concurrent processing of multiple search queries"""
        queries = ["query1", "query2", "query3"]
        
        async def test():
            results = await self.processor.search_concurrent(queries)
            
            self.assertEqual(len(results), 3)
            for i, result in enumerate(results):
                self.assertIn("results", result)
                self.assertEqual(result["query"], queries[i])
                self.assertEqual(len(result["results"]), 2)
        
        asyncio.run(test())
    
    def test_single_search_async(self):
        """Test single search with async processing"""
        async def test():
            result = await self.processor.search_single("test query")
            self.assertEqual(len(result), 2)
            self.assertIn("result for test query", result[0][0])
        
        asyncio.run(test())
    
    def test_search_error_handling(self):
        """Test error handling in concurrent search"""
        def failing_search(query: str, **kwargs):
            if "fail" in query:
                raise Exception(f"Search failed for {query}")
            return [(f"result for {query}", 0.8)]
        
        processor = ConcurrentSearchProcessor(failing_search)
        queries = ["good query", "fail query", "another good"]
        
        async def test():
            results = await processor.search_concurrent(queries)
            
            self.assertEqual(len(results), 3)
            
            # Good queries should have results
            self.assertIn("results", results[0])
            self.assertIn("results", results[2])
            
            # Failed query should have error
            self.assertIn("error", results[1])
        
        asyncio.run(test())
    
    def test_processor_stats(self):
        """Test processor statistics"""
        async def test():
            await self.processor.search_single("test")
            stats = self.processor.get_stats()
            
            self.assertIn("handler_stats", stats)
            self.assertIn("cache_size", stats)
            self.assertGreater(stats["handler_stats"]["total_requests"], 0)
        
        asyncio.run(test())

class TestAsyncDecorators(unittest.TestCase):
    """Test async decorators for performance optimization"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_files = []
    
    def tearDown(self):
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_async_cached_decorator(self):
        """Test async caching decorator"""
        call_count = 0
        
        @async_cached(ttl=300)
        async def cached_func(x):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return x * 2
        
        async def test():
            nonlocal call_count
            
            # First call
            result1 = await cached_func(5)
            self.assertEqual(result1, 10)
            self.assertEqual(call_count, 1)
            
            # Second call (should be cached)
            result2 = await cached_func(5)
            self.assertEqual(result2, 10)
            self.assertEqual(call_count, 1)  # No additional call
            
            # Different argument (should call again)
            result3 = await cached_func(3)
            self.assertEqual(result3, 6)
            self.assertEqual(call_count, 2)
        
        asyncio.run(test())
    
    def test_rate_limiting_decorator(self):
        """Test rate limiting decorator"""
        @rate_limit(max_calls=3, window=1)
        async def rate_limited_func():
            return "success"
        
        async def test():
            # First 3 calls should succeed
            for _ in range(3):
                result = await rate_limited_func()
                self.assertEqual(result, "success")
            
            # 4th call should fail
            with self.assertRaises(Exception) as context:
                await rate_limited_func()
            
            self.assertIn("Rate limit exceeded", str(context.exception))
        
        asyncio.run(test())

class TestGlobalHandler(unittest.TestCase):
    """Test global async handler functionality"""
    
    def test_global_instance(self):
        """Test global handler singleton"""
        handler1 = get_async_handler()
        handler2 = get_async_handler()
        self.assertIs(handler1, handler2)
    
    def test_concurrent_search_utility(self):
        """Test utility concurrent search function"""
        def mock_search(query: str, **kwargs):
            return [(f"result for {query}", 0.8)]
        
        async def test():
            queries = ["query1", "query2"]
            results = await concurrent_search(mock_search, queries)
            
            self.assertEqual(len(results), 2)
            for result in results:
                self.assertIn("results", result)
        
        asyncio.run(test())

class TestAsyncPerformance(unittest.TestCase):
    """Test async handler performance characteristics"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_files = []
    
    def tearDown(self):
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_concurrent_performance(self):
        """Test performance with high concurrency"""
        handler = AsyncRequestHandler(max_workers=10, queue_size=100)
        
        def compute_task(n):
            time.sleep(0.01)  # Simulate work
            return n * 2
        
        async def test():
            start_time = time.time()
            
            # Process 50 tasks concurrently
            tasks = [handler.handle_request(compute_task, i) for i in range(50)]
            results = await asyncio.gather(*tasks)
            
            elapsed = time.time() - start_time
            
            # Verify results
            self.assertEqual(len(results), 50)
            for i, result in enumerate(results):
                self.assertEqual(result, i * 2)
            
            # Should be faster than sequential execution
            self.assertLess(elapsed, 0.8)  # Should complete in less than 0.8s
            
            # Check stats
            stats = handler.get_stats()
            self.assertEqual(stats["total_requests"], 50)
            self.assertGreater(stats["throughput_rps"], 25)  # At least 25 RPS (more realistic)
        
        try:
            asyncio.run(test())
        finally:
            handler.shutdown()
    
    def test_search_processor_performance(self):
        """Test search processor performance with realistic workload"""
        def realistic_search(query: str, **kwargs):
            # Simulate variable search time
            time.sleep(0.02 + len(query) * 0.001)
            return [(f"result 1 for {query}", 0.9), (f"result 2 for {query}", 0.7)]
        
        processor = ConcurrentSearchProcessor(realistic_search, max_concurrent=15)
        
        async def test():
            # Generate realistic queries
            queries = [f"search query number {i}" for i in range(20)]
            
            start_time = time.time()
            results = await processor.search_concurrent(queries)
            elapsed = time.time() - start_time
            
            # Verify all results
            self.assertEqual(len(results), 20)
            for result in results:
                self.assertIn("results", result)
                self.assertEqual(len(result["results"]), 2)
            
            # Should complete reasonably fast
            self.assertLess(elapsed, 1.5)  # Under 1.5 seconds
            
            # Check performance stats
            stats = processor.get_stats()
            handler_stats = stats["handler_stats"]
            self.assertEqual(handler_stats["total_requests"], 20)
        
        asyncio.run(test())
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large batches"""
        handler = AsyncRequestHandler(max_workers=8, queue_size=200)
        
        def memory_task(data_size):
            # Create and process some data
            data = list(range(data_size))
            return sum(data)
        
        async def test():
            # Process many tasks with varying data sizes
            tasks = [handler.handle_request(memory_task, i * 100) for i in range(1, 101)]
            results = await asyncio.gather(*tasks)
            
            # Verify results are correct
            self.assertEqual(len(results), 100)
            
            # Check that handler maintained reasonable stats
            stats = handler.get_stats()
            self.assertEqual(stats["total_requests"], 100)
            self.assertLess(stats["avg_response_time"], 0.1)  # Should be fast
        
        try:
            asyncio.run(test())
        finally:
            handler.shutdown()

if __name__ == "__main__":
    unittest.main() 