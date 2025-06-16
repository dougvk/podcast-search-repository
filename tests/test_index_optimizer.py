#!/usr/bin/env python3
"""
Tests for FAISS index optimizer with performance validation
"""

import unittest
import tempfile
import shutil
import os
import numpy as np
import time
from pathlib import Path

from core.index_optimizer import (
    IndexOptimizer, IndexConfig, IndexType, create_optimized_index
)

class TestIndexOptimizer(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.temp_files = []
        
        # Create test vectors
        np.random.seed(42)  # Reproducible tests
        self.small_vectors = np.random.randn(100, 384).astype(np.float32)
        self.medium_vectors = np.random.randn(5000, 384).astype(np.float32)
        self.large_vectors = np.random.randn(20000, 384).astype(np.float32)
        
        # Normalize vectors (typical for embeddings)
        for vectors in [self.small_vectors, self.medium_vectors, self.large_vectors]:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors /= np.clip(norms, 1e-8, None)
    
    def tearDown(self):
        """Clean up test environment"""
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        # Clean up any .faiss or .pkl files in current directory
        for ext in ['.faiss', '.pkl', '.pkl.gz']:
            for file in Path('.').glob(f'*{ext}'):
                file.unlink(missing_ok=True)
    
    def test_auto_configuration(self):
        """Test automatic index configuration based on dataset size"""
        optimizer = IndexOptimizer()
        
        # Test small dataset - should use Flat
        config = optimizer.auto_configure(500)
        self.assertEqual(config.index_type, IndexType.FLAT)
        
        # Test medium dataset - should use IVF_FLAT
        config = optimizer.auto_configure(5000)
        self.assertEqual(config.index_type, IndexType.IVF_FLAT)
        self.assertTrue(4 <= config.nlist <= 4096)
        
        # Test large dataset - should use IVF_PQ
        config = optimizer.auto_configure(150000)
        self.assertEqual(config.index_type, IndexType.IVF_PQ)
        self.assertTrue(1000 <= config.nlist <= 8192)
        self.assertTrue(config.m > 0)
    
    def test_index_building_performance(self):
        """Test index building with different configurations"""
        optimizer = IndexOptimizer()
        
        # Test Flat index (small dataset)
        start_time = time.time()
        stats = optimizer.build_index(self.small_vectors)
        build_time = time.time() - start_time
        
        self.assertTrue(stats["is_trained"])
        self.assertEqual(stats["ntotal"], len(self.small_vectors))
        self.assertLess(build_time, 5.0)  # Should be fast for small dataset
        
        # Test IVF index (medium dataset) 
        optimizer = IndexOptimizer()
        config = IndexConfig(index_type=IndexType.IVF_FLAT, nlist=50)
        
        start_time = time.time()
        stats = optimizer.build_index(self.medium_vectors, config)
        build_time = time.time() - start_time
        
        self.assertTrue(stats["is_trained"])
        self.assertEqual(stats["ntotal"], len(self.medium_vectors))
        self.assertLess(build_time, 10.0)  # Should be reasonable for medium dataset
    
    def test_search_functionality(self):
        """Test vector search with different index types"""
        optimizer = IndexOptimizer()
        
        # Build index
        optimizer.build_index(self.medium_vectors)
        
        # Single vector search
        query_vector = self.medium_vectors[0]
        scores, indices = optimizer.search(query_vector, k=5)
        
        self.assertEqual(len(scores[0]), 5)
        self.assertEqual(len(indices[0]), 5)
        self.assertEqual(indices[0][0], 0)  # Should find itself first
        
        # Multiple vector search
        query_vectors = self.medium_vectors[:10]
        scores, indices = optimizer.search(query_vectors, k=3)
        
        self.assertEqual(scores.shape, (10, 3))
        self.assertEqual(indices.shape, (10, 3))
    
    def test_parameter_optimization(self):
        """Test automatic parameter optimization"""
        optimizer = IndexOptimizer()
        
        # Build IVF index
        config = IndexConfig(index_type=IndexType.IVF_FLAT, nlist=100)
        optimizer.build_index(self.medium_vectors, config)
        
        # Test parameter optimization
        sample_queries = self.medium_vectors[:50]
        results = optimizer.optimize_search_params(sample_queries)
        
        self.assertIn("optimized_nprobe", results)
        self.assertIn("avg_search_time", results)
        self.assertTrue(1 <= results["optimized_nprobe"] <= 100)
        self.assertGreater(results["avg_search_time"], 0)
    
    def test_save_and_load_index(self):
        """Test index serialization and loading"""
        optimizer = IndexOptimizer()
        optimizer.build_index(self.small_vectors)
        
        # Save index
        index_path = self.temp_dir / "test_index"
        save_stats = optimizer.save_index(index_path, compress=True)
        
        self.assertTrue(Path(save_stats["index_path"]).exists())
        self.assertTrue(Path(save_stats["metadata_path"]).exists())
        self.assertGreater(save_stats["index_size_mb"], 0)
        
        # Load index into new optimizer
        new_optimizer = IndexOptimizer()
        load_stats = new_optimizer.load_index(index_path)
        
        self.assertEqual(load_stats["ntotal"], len(self.small_vectors))
        self.assertEqual(load_stats["dimension"], 384)
        
        # Test that loaded index works
        query_vector = self.small_vectors[0]
        scores, indices = new_optimizer.search(query_vector, k=3)
        
        self.assertEqual(len(scores[0]), 3)
        self.assertEqual(indices[0][0], 0)  # Should find itself first
    
    def test_stats_tracking(self):
        """Test performance statistics tracking"""
        optimizer = IndexOptimizer()
        
        # Initially no stats
        stats = optimizer.get_stats()
        self.assertEqual(stats["builds"], 0)
        self.assertEqual(stats["searches"], 0)
        
        # Build index
        optimizer.build_index(self.small_vectors)
        stats = optimizer.get_stats()
        self.assertEqual(stats["builds"], 1)
        self.assertGreater(stats["optimization_time"], 0)
        
        # Perform searches
        query_vector = self.small_vectors[0]
        for _ in range(5):
            optimizer.search(query_vector, k=3)
        
        stats = optimizer.get_stats()
        self.assertEqual(stats["searches"], 5)
        self.assertIn("ntotal", stats)
    
    def test_factory_function(self):
        """Test the create_optimized_index factory function"""
        # Test with automatic configuration
        optimizer = create_optimized_index(self.medium_vectors)
        
        self.assertIsNotNone(optimizer.index)
        self.assertEqual(optimizer.index.ntotal, len(self.medium_vectors))
        
        # Test search functionality
        query_vector = self.medium_vectors[0]
        scores, indices = optimizer.search(query_vector, k=5)
        
        self.assertEqual(len(scores[0]), 5)
        self.assertEqual(indices[0][0], 0)
    
    def test_large_dataset_optimization(self):
        """Test optimization strategies for large datasets"""
        # Create larger test dataset
        large_vectors = np.random.randn(50000, 384).astype(np.float32)
        norms = np.linalg.norm(large_vectors, axis=1, keepdims=True)
        large_vectors /= np.clip(norms, 1e-8, None)
        
        # Test with target size (should use sampling)
        optimizer = create_optimized_index(large_vectors, target_size=10000)
        
        # Verify it built successfully
        self.assertIsNotNone(optimizer.index)
        stats = optimizer.get_stats()
        self.assertGreater(stats["ntotal"], 0)
        
        # Test search performance
        query_vector = large_vectors[0]
        start_time = time.time()
        scores, indices = optimizer.search(query_vector, k=10)
        search_time = time.time() - start_time
        
        self.assertLess(search_time, 1.0)  # Should be fast even for large dataset
        self.assertEqual(len(scores[0]), 10)

class TestIndexOptimizerIntegration(unittest.TestCase):
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create realistic embedding-like data
        np.random.seed(42)
        self.embeddings = np.random.randn(1000, 384).astype(np.float32)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings /= np.clip(norms, 1e-8, None)
    
    def tearDown(self):
        """Clean up integration test environment"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        # Clean up any artifacts
        for ext in ['.faiss', '.pkl', '.pkl.gz']:
            for file in Path('.').glob(f'*{ext}'):
                file.unlink(missing_ok=True)
    
    def test_end_to_end_optimization_workflow(self):
        """Test complete optimization workflow"""
        optimizer = IndexOptimizer()
        
        # Step 1: Auto-configure
        config = optimizer.auto_configure(len(self.embeddings), self.embeddings[:100])
        self.assertIsInstance(config, IndexConfig)
        
        # Step 2: Build optimized index
        build_stats = optimizer.build_index(self.embeddings, config)
        self.assertTrue(build_stats["is_trained"])
        self.assertEqual(build_stats["ntotal"], len(self.embeddings))
        
        # Step 3: Optimize search parameters
        if hasattr(optimizer.index, 'nprobe'):
            optimization_results = optimizer.optimize_search_params(self.embeddings[:50])
            self.assertIn("optimized_nprobe", optimization_results)
        
        # Step 4: Performance validation
        query_vectors = self.embeddings[:10]
        start_time = time.time()
        scores, indices = optimizer.search(query_vectors, k=5)
        search_time = time.time() - start_time
        
        self.assertLess(search_time, 0.1)  # Sub-100ms for 10 queries
        self.assertEqual(scores.shape, (10, 5))
        
        # Step 5: Save optimized index
        index_path = self.temp_dir / "optimized_index"
        save_stats = optimizer.save_index(index_path)
        self.assertTrue(Path(save_stats["index_path"]).exists())

if __name__ == "__main__":
    unittest.main() 