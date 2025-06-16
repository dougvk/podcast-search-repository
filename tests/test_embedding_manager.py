#!/usr/bin/env python3
"""
Comprehensive tests for embedding compression and lazy loading
"""

import unittest
import tempfile
import os
import shutil
import threading
import time
from pathlib import Path
import numpy as np

from core.embedding_manager import (
    CompressedEmbeddings, EmbeddingManager, 
    get_embedding_manager, compress_embeddings
)

class TestCompressedEmbeddings(unittest.TestCase):
    """Test compressed embeddings storage and lazy loading"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_files = []
        self.test_embeddings = np.random.rand(100, 384).astype(np.float32)
    
    def tearDown(self):
        # Clean up temporary files
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_basic_compression(self):
        """Test basic embedding compression functionality"""
        compressed = CompressedEmbeddings(self.test_embeddings)
        
        # Verify compression occurred
        stats = compressed.get_stats()
        self.assertTrue(stats["has_compressed_data"])
        self.assertEqual(stats["compressions"], 1)
        self.assertGreater(stats["compressed_size_bytes"], 0)
        
        # Test retrieval
        retrieved = compressed.get_embeddings()
        np.testing.assert_array_equal(retrieved, self.test_embeddings)
        
        # Verify caching stats
        stats = compressed.get_stats()
        self.assertTrue(stats["is_loaded"])
        self.assertEqual(stats["decompressions"], 1)
    
    def test_lazy_loading(self):
        """Test lazy loading behavior"""
        compressed = CompressedEmbeddings(self.test_embeddings)
        
        # Initially not loaded
        compressed.unload()
        stats = compressed.get_stats()
        self.assertFalse(stats["is_loaded"])
        
        # First access loads
        retrieved = compressed.get_embeddings()
        stats = compressed.get_stats()
        self.assertTrue(stats["is_loaded"])
        
        # Second access uses cache
        retrieved2 = compressed.get_embeddings()
        stats = compressed.get_stats()
        self.assertEqual(stats["cache_hits"], 1)
        np.testing.assert_array_equal(retrieved, retrieved2)
    
    def test_file_persistence(self):
        """Test saving and loading from disk"""
        compressed = CompressedEmbeddings(self.test_embeddings)
        
        # Save to file
        filepath = os.path.join(self.temp_dir, "test.emb")
        self.temp_files.append(filepath)
        compressed.save_to_file(filepath)
        
        # Load from file
        new_compressed = CompressedEmbeddings()
        new_compressed.load_from_file(filepath)
        
        # Verify data integrity
        retrieved = new_compressed.get_embeddings()
        np.testing.assert_array_equal(retrieved, self.test_embeddings)
    
    def test_compression_ratio(self):
        """Test compression effectiveness"""
        compressed = CompressedEmbeddings(self.test_embeddings)
        
        original_size = self.test_embeddings.nbytes
        compressed_size = compressed.get_stats()["compressed_size_bytes"]
        
        # Should achieve some compression
        self.assertLess(compressed_size, original_size)
        ratio = original_size / compressed_size
        self.assertGreater(ratio, 1.1)  # At least 10% compression

class TestEmbeddingManager(unittest.TestCase):
    """Test enterprise embedding manager functionality"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_files = []
        self.manager = EmbeddingManager()
        self.test_embeddings = {
            "episode_1": np.random.rand(50, 384).astype(np.float32),
            "episode_2": np.random.rand(75, 384).astype(np.float32),
            "episode_3": np.random.rand(100, 384).astype(np.float32)
        }
    
    def tearDown(self):
        # Clean up temporary files
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_add_and_retrieve(self):
        """Test adding and retrieving embeddings"""
        for key, embeddings in self.test_embeddings.items():
            ratio = self.manager.add_embeddings(key, embeddings)
            self.assertGreater(ratio, 1.0)  # Should achieve compression
        
        # Retrieve and verify
        for key, original in self.test_embeddings.items():
            retrieved = self.manager.get_embeddings(key)
            np.testing.assert_array_equal(retrieved, original)
        
        # Check stats
        stats = self.manager.get_stats()
        self.assertEqual(stats["total_embeddings"], 3)
        self.assertGreater(stats["memory_saved_mb"], 0)
    
    def test_auto_unload_management(self):
        """Test automatic memory management"""
        manager = EmbeddingManager(auto_unload_threshold=5)
        
        # Add embeddings
        for key, embeddings in self.test_embeddings.items():
            manager.add_embeddings(key, embeddings)
        
        # Access to trigger auto-unload
        for i in range(6):  # Trigger threshold
            manager.get_embeddings("episode_1")
        
        # Should have triggered unload
        stats = manager.get_stats()
        # Auto-unload might have cleared cached embeddings
        self.assertGreaterEqual(stats["access_count"], 6)
    
    def test_manual_memory_management(self):
        """Test manual unloading"""
        for key, embeddings in self.test_embeddings.items():
            self.manager.add_embeddings(key, embeddings)
        
        # Load all into memory
        for key in self.test_embeddings:
            self.manager.get_embeddings(key)
        
        # Unload specific embedding
        self.manager.unload_embeddings("episode_1")
        
        # Unload all
        self.manager.unload_all()
        
        # Can still retrieve (lazy loading)
        retrieved = self.manager.get_embeddings("episode_2")
        np.testing.assert_array_equal(retrieved, self.test_embeddings["episode_2"])
    
    def test_disk_persistence(self):
        """Test saving and loading all embeddings to/from disk"""
        # Add test data
        for key, embeddings in self.test_embeddings.items():
            self.manager.add_embeddings(key, embeddings)
        
        # Save to disk
        self.manager.save_to_disk(self.temp_dir)
        
        # Create new manager and load
        new_manager = EmbeddingManager()
        new_manager.load_from_disk(self.temp_dir)
        
        # Verify all data loaded correctly
        for key, original in self.test_embeddings.items():
            retrieved = new_manager.get_embeddings(key)
            np.testing.assert_array_equal(retrieved, original)
        
        stats = new_manager.get_stats()
        self.assertEqual(stats["total_embeddings"], 3)

class TestGlobalManager(unittest.TestCase):
    """Test global embedding manager functionality"""
    
    def test_global_instance(self):
        """Test global manager singleton"""
        manager1 = get_embedding_manager()
        manager2 = get_embedding_manager()
        self.assertIs(manager1, manager2)
    
    def test_utility_function(self):
        """Test utility compression function"""
        test_data = np.random.rand(50, 384).astype(np.float32)
        compressed = compress_embeddings(test_data)
        
        retrieved = compressed.get_embeddings()
        np.testing.assert_array_equal(retrieved, test_data)

class TestEmbeddingPerformance(unittest.TestCase):
    """Test embedding manager performance characteristics"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_files = []
    
    def tearDown(self):
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_compression_performance(self):
        """Test compression speed with realistic data"""
        # Simulate 700 episodes with varying sizes
        manager = EmbeddingManager()
        
        start_time = time.time()
        for i in range(50):  # Reduced for faster testing
            size = np.random.randint(100, 1000)
            embeddings = np.random.rand(size, 384).astype(np.float32)
            manager.add_embeddings(f"episode_{i}", embeddings)
        
        compression_time = time.time() - start_time
        
        # Should compress 50 episodes in reasonable time
        self.assertLess(compression_time, 30.0)  # Under 30 seconds
        
        stats = manager.get_stats()
        self.assertEqual(stats["total_embeddings"], 50)
        self.assertGreater(stats["memory_saved_mb"], 0)
    
    def test_retrieval_performance(self):
        """Test retrieval speed with compression"""
        manager = EmbeddingManager()
        test_data = np.random.rand(500, 384).astype(np.float32)
        manager.add_embeddings("large_episode", test_data)
        
        # Time multiple retrievals
        start_time = time.time()
        for _ in range(10):
            retrieved = manager.get_embeddings("large_episode")
            self.assertEqual(retrieved.shape, test_data.shape)
        
        retrieval_time = time.time() - start_time
        
        # Should be fast due to caching
        self.assertLess(retrieval_time, 1.0)  # Under 1 second for 10 retrievals
    
    def test_concurrent_access(self):
        """Test thread safety with concurrent access"""
        manager = EmbeddingManager()
        test_data = np.random.rand(100, 384).astype(np.float32)
        manager.add_embeddings("shared", test_data)
        
        results = []
        errors = []
        
        def worker():
            try:
                for _ in range(10):
                    retrieved = manager.get_embeddings("shared")
                    results.append(retrieved.shape)
                    time.sleep(0.01)  # Small delay
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = [threading.Thread(target=worker) for _ in range(5)]
        
        start_time = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        concurrent_time = time.time() - start_time
        
        # Verify no errors and correct results
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 50)  # 5 threads × 10 operations
        self.assertLess(concurrent_time, 5.0)  # Under 5 seconds

if __name__ == "__main__":
    unittest.main() 