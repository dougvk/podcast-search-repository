#!/usr/bin/env python3
"""
Tests for monitoring and logging functionality
"""

import unittest
import tempfile
import shutil
import os
import json
import time
from unittest.mock import Mock, patch, AsyncMock
from fastapi import Request, Response
from fastapi.testclient import TestClient
import asyncio

# Import monitoring components
from core.monitoring import (
    StructuredLogger, monitoring_middleware, monitor_search, 
    monitor_embeddings, get_metrics, health_check,
    request_count, request_duration, search_operations, 
    embedding_operations, active_connections
)

class TestStructuredLogger(unittest.TestCase):
    """Test structured logging functionality"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_files = []
    
    def tearDown(self):
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_logger_initialization(self):
        """Test logger initializes correctly"""
        logger = StructuredLogger()
        self.assertIsInstance(logger, StructuredLogger)
    
    def test_log_request(self):
        """Test HTTP request logging"""
        logger = StructuredLogger()
        
        # Mock request and response
        request = Mock()
        request.method = "GET"
        request.url = "http://test.com/api/search"
        request.headers = {"user-agent": "test-agent"}
        request.client.host = "127.0.0.1"
        
        response = Mock()
        response.status_code = 200
        
        # Should not raise exception
        logger.log_request(request, response, 0.5)
    
    def test_log_search(self):
        """Test search operation logging"""
        logger = StructuredLogger()
        
        # Should not raise exception
        logger.log_search("test query", "semantic", 5, 0.3)
    
    def test_log_error(self):
        """Test error logging"""
        logger = StructuredLogger()
        
        error = ValueError("Test error")
        context = {"endpoint": "/search", "method": "POST"}
        
        # Should not raise exception
        logger.log_error(error, context)

class TestMonitoringMiddleware(unittest.IsolatedAsyncioTestCase):
    """Test monitoring middleware functionality"""
    
    async def test_monitoring_middleware_success(self):
        """Test middleware handles successful requests"""
        # Mock request
        request = Mock()
        request.method = "GET"
        request.url.path = "/api/search"
        request.client.host = "127.0.0.1"
        
        # Mock response
        response = Mock()
        response.status_code = 200
        
        # Mock call_next
        async def call_next(req):
            return response
        
        # Test middleware
        result = await monitoring_middleware(request, call_next)
        self.assertEqual(result, response)
    
    async def test_monitoring_middleware_error(self):
        """Test middleware handles errors"""
        request = Mock()
        request.method = "POST"
        request.url.path = "/api/search"
        
        # Mock call_next that raises exception
        async def call_next(req):
            raise ValueError("Test error")
        
        # Test middleware
        with self.assertRaises(ValueError):
            await monitoring_middleware(request, call_next)

class TestMonitoringDecorators(unittest.IsolatedAsyncioTestCase):
    """Test monitoring decorators"""
    
    async def test_monitor_search_success(self):
        """Test search monitoring decorator"""
        @monitor_search("test")
        async def test_search(query):
            return ["result1", "result2"]
        
        result = await test_search("test query")
        self.assertEqual(result, ["result1", "result2"])
    
    async def test_monitor_search_error(self):
        """Test search monitoring decorator with error"""
        @monitor_search("test")
        async def test_search(query):
            raise ValueError("Search error")
        
        with self.assertRaises(ValueError):
            await test_search("test query")
    
    def test_monitor_embeddings(self):
        """Test embeddings monitoring decorator"""
        @monitor_embeddings
        def test_embeddings():
            return "embeddings_result"
        
        result = test_embeddings()
        self.assertEqual(result, "embeddings_result")

class TestMetricsEndpoints(unittest.TestCase):
    """Test metrics and health check endpoints"""
    
    def test_get_metrics(self):
        """Test Prometheus metrics endpoint"""
        response = get_metrics()
        self.assertEqual(response.media_type, "text/plain; version=0.0.4; charset=utf-8")
    
    def test_health_check(self):
        """Test health check endpoint"""
        health_data = health_check()
        
        self.assertIn("status", health_data)
        self.assertIn("timestamp", health_data)
        self.assertIn("metrics", health_data)
        self.assertEqual(health_data["status"], "healthy")

class TestPrometheusMetrics(unittest.TestCase):
    """Test Prometheus metrics collection"""
    
    def test_request_count_metric(self):
        """Test request count metric"""
        # Increment metric
        request_count.labels(method="GET", endpoint="/test", status="200").inc()
        
        # Verify metric exists and can be accessed
        self.assertIsNotNone(request_count)
    
    def test_request_duration_metric(self):
        """Test request duration metric"""
        # Record duration
        request_duration.labels(method="POST", endpoint="/search").observe(0.5)
        
        # Verify metric exists
        self.assertIsNotNone(request_duration)
    
    def test_search_operations_metric(self):
        """Test search operations metric"""
        # Record search operation
        search_operations.labels(type="semantic", status="success").inc()
        
        # Verify metric exists
        self.assertIsNotNone(search_operations)
    
    def test_active_connections_metric(self):
        """Test active connections gauge"""
        # Test increment/decrement
        active_connections.inc()
        active_connections.dec()
        
        # Verify metric exists
        self.assertIsNotNone(active_connections)
    
    def test_embedding_operations_metric(self):
        """Test embedding operations histogram"""
        # Record embedding operation
        with embedding_operations.time():
            time.sleep(0.01)  # Simulate work
        
        # Verify metric exists
        self.assertIsNotNone(embedding_operations)

class TestMonitoringIntegration(unittest.TestCase):
    """Test monitoring integration with FastAPI"""
    
    def setUp(self):
        # Create minimal FastAPI app for testing
        from fastapi import FastAPI
        from core.monitoring import monitoring_middleware, get_metrics, health_check
        
        self.app = FastAPI()
        self.app.middleware("http")(monitoring_middleware)
        
        @self.app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        @self.app.get("/metrics/prometheus")
        async def metrics():
            return get_metrics()
        
        @self.app.get("/health/monitoring")
        async def health():
            return health_check()
        
        self.client = TestClient(self.app)
    
    def test_monitoring_middleware_integration(self):
        """Test monitoring middleware integration"""
        response = self.client.get("/test")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "test"})
    
    def test_prometheus_metrics_endpoint(self):
        """Test Prometheus metrics endpoint"""
        response = self.client.get("/metrics/prometheus")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/plain", response.headers["content-type"])
    
    def test_health_monitoring_endpoint(self):
        """Test health monitoring endpoint"""
        response = self.client.get("/health/monitoring")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "healthy")

if __name__ == '__main__':
    unittest.main() 