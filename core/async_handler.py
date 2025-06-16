"""
High-Performance Async Request Handler

Minimal code for maximum concurrent request handling efficiency.
Follows memvid patterns with thread pools and async processing.
"""

import asyncio
import threading
import time
import logging
from typing import Dict, List, Optional, Any, Callable, Awaitable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
import weakref

logger = logging.getLogger(__name__)

class AsyncRequestHandler:
    """Enterprise-grade async request handler with thread pooling"""
    
    def __init__(self, max_workers: int = 10, queue_size: int = 1000):
        self._lock = threading.RLock()
        self._max_workers = max_workers
        self._queue_size = queue_size
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._semaphore = asyncio.Semaphore(queue_size)
        self._active_tasks = weakref.WeakSet()
        self._stats = {"requests": 0, "concurrent_peak": 0, "total_time": 0.0}
    
    async def handle_request(self, func: Callable, *args, **kwargs) -> Any:
        """Handle request with async concurrency and resource limiting"""
        async with self._semaphore:
            start_time = time.time()
            
            with self._lock:
                self._stats["requests"] += 1
                current_active = len(self._active_tasks)
                if current_active > self._stats["concurrent_peak"]:
                    self._stats["concurrent_peak"] = current_active
            
            try:
                # Create task for tracking
                task = asyncio.current_task()
                if task:
                    self._active_tasks.add(task)
                
                # Run in thread pool if sync function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(self._executor, func, *args, **kwargs)
                
                elapsed = time.time() - start_time
                with self._lock:
                    self._stats["total_time"] += elapsed
                
                return result
                
            except Exception as e:
                logger.error(f"Request handling error: {e}")
                raise
    
    async def batch_requests(self, requests: List[tuple]) -> List[Any]:
        """Process multiple requests concurrently"""
        tasks = [self.handle_request(func, *args, **kwargs) for func, args, kwargs in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get handler performance statistics"""
        with self._lock:
            active_count = len(self._active_tasks)
            avg_time = self._stats["total_time"] / max(self._stats["requests"], 1)
            
            return {
                "active_requests": active_count,
                "total_requests": self._stats["requests"],
                "concurrent_peak": self._stats["concurrent_peak"],
                "avg_response_time": avg_time,
                "throughput_rps": self._stats["requests"] / max(self._stats["total_time"], 1),
                "executor_workers": self._max_workers,
                "queue_size": self._queue_size
            }
    
    def shutdown(self):
        """Gracefully shutdown the handler"""
        self._executor.shutdown(wait=True)

class ConcurrentSearchProcessor:
    """Optimized concurrent search request processor"""
    
    def __init__(self, search_func: Callable, max_concurrent: int = 20):
        self._search_func = search_func
        self._handler = AsyncRequestHandler(max_workers=max_concurrent)
        self._request_cache = {}
        self._cache_lock = threading.RLock()
    
    async def search_concurrent(self, queries: List[str], **kwargs) -> List[Dict]:
        """Process multiple search queries concurrently"""
        requests = [(self._search_func, (query,), kwargs) for query in queries]
        results = await self._handler.batch_requests(requests)
        
        # Format results
        formatted = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                formatted.append({"error": str(result), "query": queries[i]})
            else:
                formatted.append({"results": result, "query": queries[i]})
        
        return formatted
    
    async def search_single(self, query: str, **kwargs) -> Any:
        """Process single search with async handling"""
        return await self._handler.handle_request(self._search_func, query, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics"""
        return {
            "handler_stats": self._handler.get_stats(),
            "cache_size": len(self._request_cache)
        }

# Async decorators for performance optimization
def async_cached(ttl: int = 300):
    """Decorator for async function caching"""
    cache = {}
    cache_lock = threading.RLock()
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            now = time.time()
            
            with cache_lock:
                if key in cache:
                    result, timestamp = cache[key]
                    if now - timestamp < ttl:
                        return result
                    else:
                        del cache[key]
            
            result = await func(*args, **kwargs)
            
            with cache_lock:
                cache[key] = (result, now)
                # Simple cleanup - remove oldest if cache too large
                if len(cache) > 1000:
                    oldest_key = min(cache.keys(), key=lambda k: cache[k][1])
                    del cache[oldest_key]
            
            return result
        return wrapper
    return decorator

def rate_limit(max_calls: int = 100, window: int = 60):
    """Decorator for async rate limiting"""
    calls = []
    lock = threading.RLock()
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            now = time.time()
            
            with lock:
                # Remove old calls outside window
                calls[:] = [t for t in calls if now - t < window]
                
                if len(calls) >= max_calls:
                    raise Exception(f"Rate limit exceeded: {max_calls} calls per {window}s")
                
                calls.append(now)
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Global handler instance
_global_handler: Optional[AsyncRequestHandler] = None

def get_async_handler() -> AsyncRequestHandler:
    """Get global async handler instance"""
    global _global_handler
    if _global_handler is None:
        _global_handler = AsyncRequestHandler()
    return _global_handler

async def concurrent_search(search_func: Callable, queries: List[str], **kwargs) -> List[Dict]:
    """Utility function for concurrent search processing"""
    processor = ConcurrentSearchProcessor(search_func)
    return await processor.search_concurrent(queries, **kwargs) 