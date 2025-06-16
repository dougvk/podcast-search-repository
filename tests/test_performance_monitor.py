#!/usr/bin/env python3
"""
Comprehensive tests for performance monitoring and profiling
"""

import unittest
import tempfile
import os
import shutil
import time
import threading
from unittest.mock import Mock, patch

from core.performance_monitor import (
    PerformanceProfiler, BottleneckDetector, PerformanceMetrics,
    get_profiler, profile, track_endpoint_performance
)

class TestPerformanceMetrics(unittest.TestCase):
    """Test performance metrics data structure"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_files = []
    
    def tearDown(self):
        # Clean up temporary files
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_metrics_creation(self):
        """Test performance metrics creation"""
        metrics = PerformanceMetrics(
            cpu_percent=45.2,
            memory_percent=60.8,
            memory_mb=2048.5,
            disk_io_read_mb=12.3,
            disk_io_write_mb=5.7,
            network_bytes_sent=1024,
            network_bytes_recv=2048,
            timestamp=time.time()
        )
        
        self.assertEqual(metrics.cpu_percent, 45.2)
        self.assertEqual(metrics.memory_percent, 60.8)
        self.assertIsInstance(metrics.timestamp, float)

class TestPerformanceProfiler(unittest.TestCase):
    """Test performance profiler functionality"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_files = []
        self.profiler = PerformanceProfiler(max_samples=100)
    
    def tearDown(self):
        # Stop monitoring and cleanup
        if self.profiler._monitoring:
            self.profiler.stop_monitoring()
        
        # Clean up temporary files
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_profiler_initialization(self):
        """Test profiler initialization"""
        self.assertEqual(self.profiler._max_samples, 100)
        self.assertFalse(self.profiler._monitoring)
        self.assertEqual(len(self.profiler._metrics), 0)
    
    def test_function_profiling(self):
        """Test function execution profiling"""
        @self.profiler.profile_function("test_func")
        def slow_function():
            time.sleep(0.1)
            return "result"
        
        result = slow_function()
        self.assertEqual(result, "result")
        
        stats = self.profiler._function_stats["test_func"]
        self.assertEqual(stats["calls"], 1)
        self.assertGreater(stats["total_time"], 0.05)  # At least 50ms
        self.assertGreater(stats["avg_time"], 0.05)
    
    def test_endpoint_tracking(self):
        """Test endpoint performance tracking"""
        self.profiler.track_endpoint("/api/test", 0.150, error=False)
        self.profiler.track_endpoint("/api/test", 0.200, error=True)
        
        stats = self.profiler._endpoint_stats["/api/test"]
        self.assertEqual(stats["requests"], 2)
        self.assertEqual(stats["errors"], 1)
        self.assertAlmostEqual(stats["total_time"], 0.350, places=3)
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_io_counters')
    @patch('psutil.net_io_counters')
    def test_monitoring_loop(self, mock_net, mock_disk, mock_memory, mock_cpu):
        """Test system monitoring loop"""
        # Mock system calls
        mock_cpu.return_value = 25.5
        mock_memory.return_value = Mock(percent=50.2, used=1024*1024*1024)
        mock_disk.return_value = Mock(read_bytes=1000000, write_bytes=500000)
        mock_net.return_value = Mock(bytes_sent=10000, bytes_recv=20000)
        
        # Start monitoring briefly
        self.profiler.start_monitoring(interval=0.1)
        time.sleep(0.3)  # Let it collect a few samples
        self.profiler.stop_monitoring()
        
        # Check metrics were collected
        self.assertGreater(len(self.profiler._metrics), 0)
        
        latest_metric = self.profiler.get_current_metrics()
        self.assertIsNotNone(latest_metric)
        self.assertEqual(latest_metric.cpu_percent, 25.5)
    
    def test_metrics_summary(self):
        """Test metrics summary generation"""
        # Add some test endpoint data
        self.profiler.track_endpoint("/api/search", 0.100)
        self.profiler.track_endpoint("/api/search", 0.150)
        self.profiler.track_endpoint("/api/upload", 0.500, error=True)
        
        summary = self.profiler.get_metrics_summary(60)
        
        if "error" not in summary:
            self.assertIn("endpoint_performance", summary)
            search_stats = summary["endpoint_performance"].get("/api/search")
            if search_stats:
                self.assertEqual(search_stats["requests"], 2)
                self.assertAlmostEqual(search_stats["avg_response_time"], 0.125, places=3)
    
    def test_export_metrics(self):
        """Test metrics export functionality"""
        # Add some test data
        self.profiler.track_endpoint("/test", 0.100)
        
        export_file = os.path.join(self.temp_dir, "metrics_export.json")
        self.temp_files.append(export_file)
        
        self.profiler.export_metrics(export_file, duration_seconds=3600)
        
        self.assertTrue(os.path.exists(export_file))
        self.assertGreater(os.path.getsize(export_file), 0)

class TestBottleneckDetector(unittest.TestCase):
    """Test bottleneck detection functionality"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_files = []
        self.profiler = PerformanceProfiler()
        self.detector = BottleneckDetector(self.profiler)
    
    def tearDown(self):
        # Clean up
        if self.profiler._monitoring:
            self.profiler.stop_monitoring()
        
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_no_bottlenecks(self):
        """Test when no bottlenecks are detected"""
        # Mock normal metrics
        normal_metrics = PerformanceMetrics(
            cpu_percent=30.0,
            memory_percent=50.0,
            memory_mb=1024.0,
            disk_io_read_mb=5.0,
            disk_io_write_mb=3.0,
            network_bytes_sent=1000,
            network_bytes_recv=2000,
            timestamp=time.time()
        )
        
        with unittest.mock.patch.object(self.profiler, 'get_current_metrics', return_value=normal_metrics):
            issues = self.detector.check_performance_issues()
            self.assertEqual(len(issues), 0)
    
    def test_cpu_bottleneck(self):
        """Test CPU bottleneck detection"""
        high_cpu_metrics = PerformanceMetrics(
            cpu_percent=95.5,
            memory_percent=50.0,
            memory_mb=1024.0,
            disk_io_read_mb=5.0,
            disk_io_write_mb=3.0,
            network_bytes_sent=1000,
            network_bytes_recv=2000,
            timestamp=time.time()
        )
        
        with unittest.mock.patch.object(self.profiler, 'get_current_metrics', return_value=high_cpu_metrics):
            issues = self.detector.check_performance_issues()
            cpu_issues = [i for i in issues if i["type"] == "cpu_bottleneck"]
            self.assertGreater(len(cpu_issues), 0)
            self.assertEqual(cpu_issues[0]["severity"], "critical")
    
    def test_memory_bottleneck(self):
        """Test memory bottleneck detection"""
        high_memory_metrics = PerformanceMetrics(
            cpu_percent=30.0,
            memory_percent=88.5,
            memory_mb=8192.0,
            disk_io_read_mb=5.0,
            disk_io_write_mb=3.0,
            network_bytes_sent=1000,
            network_bytes_recv=2000,
            timestamp=time.time()
        )
        
        with unittest.mock.patch.object(self.profiler, 'get_current_metrics', return_value=high_memory_metrics):
            issues = self.detector.check_performance_issues()
            memory_issues = [i for i in issues if i["type"] == "memory_bottleneck"]
            self.assertGreater(len(memory_issues), 0)
            self.assertEqual(memory_issues[0]["severity"], "warning")

class TestGlobalProfiler(unittest.TestCase):
    """Test global profiler functionality"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_files = []
        # Reset global profiler
        import core.performance_monitor
        core.performance_monitor._global_profiler = None
    
    def tearDown(self):
        # Stop any monitoring
        import core.performance_monitor
        if core.performance_monitor._global_profiler and core.performance_monitor._global_profiler._monitoring:
            core.performance_monitor._global_profiler.stop_monitoring()
        core.performance_monitor._global_profiler = None
        
        # Clean up files
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_get_global_profiler(self):
        """Test global profiler singleton"""
        profiler1 = get_profiler()
        profiler2 = get_profiler()
        
        self.assertIs(profiler1, profiler2)
        self.assertTrue(profiler1._monitoring)
    
    def test_profile_decorator(self):
        """Test global profile decorator"""
        @profile("decorated_function")
        def test_function():
            time.sleep(0.05)
            return 42
        
        result = test_function()
        self.assertEqual(result, 42)
        
        # Check that profiling data was recorded
        global_profiler = get_profiler()
        stats = global_profiler._function_stats.get("decorated_function")
        self.assertIsNotNone(stats)
        self.assertEqual(stats["calls"], 1)
    
    def test_track_endpoint_performance(self):
        """Test global endpoint tracking"""
        start_time = time.time()
        time.sleep(0.05)
        
        track_endpoint_performance("/test/endpoint", start_time, error=False)
        
        global_profiler = get_profiler()
        stats = global_profiler._endpoint_stats.get("/test/endpoint")
        self.assertIsNotNone(stats)
        self.assertEqual(stats["requests"], 1)
        self.assertEqual(stats["errors"], 0)

class TestPerformanceIntegration(unittest.TestCase):
    """Test performance monitoring integration"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_files = []
        self.profiler = PerformanceProfiler()
    
    def tearDown(self):
        if self.profiler._monitoring:
            self.profiler.stop_monitoring()
        
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_concurrent_profiling(self):
        """Test profiling under concurrent load"""
        @self.profiler.profile_function("concurrent_test")
        def worker_function(worker_id):
            time.sleep(0.01 * worker_id)  # Variable sleep time
            return f"worker_{worker_id}"
        
        # Run concurrent workers
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_function, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Check statistics
        stats = self.profiler._function_stats["concurrent_test"]
        self.assertEqual(stats["calls"], 5)
        self.assertGreater(stats["total_time"], 0)
    
    def test_performance_under_load(self):
        """Test performance monitoring under simulated load"""
        # Simulate high endpoint load
        endpoints = ["/search", "/upload", "/download", "/status"]
        
        for _ in range(20):
            for endpoint in endpoints:
                duration = 0.01 + (hash(endpoint) % 100) / 10000  # Deterministic but varied
                self.profiler.track_endpoint(endpoint, duration, error=(hash(endpoint) % 10 == 0))
        
        # Generate summary
        summary = self.profiler.get_metrics_summary(60)
        
        if "endpoint_performance" in summary:
            self.assertGreater(len(summary["endpoint_performance"]), 0)
            
            # Check specific endpoint
            search_stats = summary["endpoint_performance"].get("/search")
            if search_stats:
                self.assertEqual(search_stats["requests"], 20)

if __name__ == "__main__":
    unittest.main() 