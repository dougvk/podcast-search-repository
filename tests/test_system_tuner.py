#!/usr/bin/env python3
"""
Comprehensive tests for system tuning and bottleneck resolution
"""

import unittest
import tempfile
import os
import shutil
import time
import threading
from unittest.mock import Mock, patch

from core.system_tuner import (
    SystemTuner, SystemConfig, get_system_tuner,
    auto_tune_system, optimize_for_search
)
from core.performance_monitor import PerformanceProfiler, PerformanceMetrics

class TestSystemConfig(unittest.TestCase):
    """Test system configuration data structure"""
    
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
        
        # Clean up any .faiss or .pkl files
        for ext in ['.faiss', '.pkl']:
            for file in os.listdir('.'):
                if file.endswith(ext):
                    try:
                        os.remove(file)
                    except:
                        pass
    
    def test_system_config_creation(self):
        """Test SystemConfig data structure"""
        config = SystemConfig(
            max_workers=8,
            memory_limit_mb=2048,
            gc_threshold=0.8,
            batch_size=32,
            cache_size_mb=512,
            embedding_compression=True,
            index_type="IVF",
            concurrent_requests=10
        )
        
        self.assertEqual(config.max_workers, 8)
        self.assertEqual(config.memory_limit_mb, 2048)
        self.assertTrue(config.embedding_compression)

class TestSystemTuner(unittest.TestCase):
    """Test system tuner functionality"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_files = []
        self.mock_profiler = Mock()
        self.tuner = SystemTuner(self.mock_profiler)
    
    def tearDown(self):
        # Clean up temporary files
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        # Clean up any .faiss or .pkl files
        for ext in ['.faiss', '.pkl']:
            for file in os.listdir('.'):
                if file.endswith(ext):
                    try:
                        os.remove(file)
                    except:
                        pass
    
    def test_get_current_config(self):
        """Test getting current system configuration"""
        config = self.tuner.get_current_config()
        self.assertIsInstance(config, SystemConfig)
        self.assertGreater(config.max_workers, 0)
    
    def test_auto_tune_basic(self):
        """Test basic auto-tuning functionality"""
        # Mock metrics
        metrics = {
            "avg_cpu_percent": 70.0,
            "avg_memory_percent": 60.0,
            "endpoint_stats": {"/search": {"avg_response_time": 500}}
        }
        bottlenecks = [{"type": "cpu", "severity": "medium"}]
        
        result = self.tuner.auto_tune(metrics, bottlenecks)
        self.assertIsInstance(result, dict)
        self.assertIn("adjustments", result)
        self.assertIn("new_config", result)
    
    def test_optimize_for_search(self):
        """Test search-specific optimization"""
        result = self.tuner.optimize_for_search()
        self.assertIsInstance(result, dict)
        self.assertIn("workload_type", result)
        self.assertIn("optimized_config", result)
        self.assertEqual(result["workload_type"], "search")
    
    def test_calculate_health_score(self):
        """Test system health score calculation"""
        # Create mock metrics
        metrics = PerformanceMetrics(
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_mb=1024.0,
            disk_io_read_mb=10.0,
            disk_io_write_mb=5.0,
            network_bytes_sent=1000.0,
            network_bytes_recv=2000.0,
            timestamp=time.time()
        )
        
        bottlenecks = []
        health_score = self.tuner.calculate_health_score(metrics, bottlenecks)
        
        self.assertIsInstance(health_score, float)
        self.assertGreaterEqual(health_score, 0.0)
        self.assertLessEqual(health_score, 1.0)
    
    def test_estimate_performance_gain(self):
        """Test performance gain estimation"""
        gain = self.tuner.estimate_performance_gain()
        self.assertIsInstance(gain, dict)
        self.assertIn("search_speedup", gain)

class TestSystemTunerIntegration(unittest.TestCase):
    """Test system tuner integration functionality"""
    
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
        
        # Clean up any .faiss or .pkl files
        for ext in ['.faiss', '.pkl']:
            for file in os.listdir('.'):
                if file.endswith(ext):
                    try:
                        os.remove(file)
                    except:
                        pass
    
    def test_get_system_tuner_singleton(self):
        """Test system tuner singleton pattern"""
        tuner1 = get_system_tuner()
        tuner2 = get_system_tuner()
        self.assertIs(tuner1, tuner2)
    
    def test_auto_tune_system_function(self):
        """Test standalone auto-tune function"""
        with patch('core.system_tuner.get_system_tuner') as mock_get_tuner:
            mock_tuner = Mock()
            mock_tuner.auto_tune.return_value = {"adjustments": ["optimization1", "optimization2"]}
            mock_get_tuner.return_value = mock_tuner
            
            result = auto_tune_system()
            self.assertIsInstance(result, dict)
    
    def test_optimize_for_search_function(self):
        """Test search optimization function"""
        mock_tuner = Mock()
        mock_tuner.optimize_for_workload.return_value = {"workload_type": "search"}
        mock_profiler = Mock()
        
        result = optimize_for_search(mock_tuner, mock_profiler)
        self.assertIsInstance(result, dict)
        mock_tuner.optimize_for_workload.assert_called_once_with("search")

if __name__ == '__main__':
    unittest.main() 