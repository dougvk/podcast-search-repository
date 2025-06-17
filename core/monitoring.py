#!/usr/bin/env python3
"""
Production Monitoring & Logging
Minimal lines, maximum observability
"""

import time
import json
import logging
from typing import Dict, Any
from functools import wraps
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Request, Response
from loguru import logger
import sys

# Prometheus Metrics
request_count = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
search_operations = Counter('search_operations_total', 'Total search operations', ['type', 'status'])
embedding_operations = Histogram('embedding_operations_seconds', 'Embedding operation duration')
active_connections = Gauge('active_connections', 'Active connections')

class StructuredLogger:
    """Minimal structured JSON logger"""
    
    def __init__(self):
        # Remove default logger and add structured JSON format
        logger.remove()
        logger.add(
            sys.stdout,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}",
            serialize=True,  # JSON output
            level="INFO"
        )
        logger.add(
            "logs/app.log",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}",
            serialize=True,
            rotation="100 MB",
            retention="30 days",
            level="INFO"
        )
    
    def log_request(self, request: Request, response: Response, duration: float):
        """Log HTTP request with structured data"""
        logger.info(
            "HTTP Request",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            duration=duration,
            user_agent=request.headers.get("user-agent", ""),
            remote_addr=request.client.host if request.client else ""
        )
    
    def log_search(self, query: str, search_type: str, results: int, duration: float):
        """Log search operation"""
        logger.info(
            "Search Operation",
            query=query[:100],  # Truncate for privacy
            search_type=search_type,
            results_count=results,
            duration=duration
        )
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with context"""
        logger.error(
            "Application Error",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context or {}
        )

# Global logger instance
structured_logger = StructuredLogger()

async def monitoring_middleware(request: Request, call_next):
    """FastAPI middleware for monitoring and logging"""
    start_time = time.time()
    active_connections.inc()
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        # Update metrics
        endpoint = request.url.path
        request_count.labels(
            method=request.method,
            endpoint=endpoint,
            status=response.status_code
        ).inc()
        request_duration.labels(
            method=request.method,
            endpoint=endpoint
        ).observe(duration)
        
        # Log request
        structured_logger.log_request(request, response, duration)
        
        return response
    
    except Exception as e:
        duration = time.time() - start_time
        structured_logger.log_error(e, {"endpoint": request.url.path, "method": request.method})
        raise
    finally:
        active_connections.dec()

def monitor_search(search_type: str):
    """Decorator to monitor search operations"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Update metrics
                search_operations.labels(type=search_type, status="success").inc()
                
                # Log search
                query = kwargs.get('query', args[0] if args else "")
                results_count = len(result) if isinstance(result, list) else 1
                structured_logger.log_search(str(query), search_type, results_count, duration)
                
                return result
            except Exception as e:
                search_operations.labels(type=search_type, status="error").inc()
                structured_logger.log_error(e, {"search_type": search_type})
                raise
        return wrapper
    return decorator

def monitor_embeddings(func):
    """Decorator to monitor embedding operations"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with embedding_operations.time():
            return func(*args, **kwargs)
    return wrapper

def get_metrics():
    """Get Prometheus metrics in text format"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Health check with metrics
def health_check():
    """Enhanced health check with basic metrics"""
    try:
        active_conn = active_connections._value._value
    except:
        active_conn = 0
    
    try:
        total_req = sum(m._value._value for m in request_count._metrics.values()) if request_count._metrics else 0
    except:
        total_req = 0
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "metrics": {
            "active_connections": active_conn,
            "total_requests": total_req
        }
    } 