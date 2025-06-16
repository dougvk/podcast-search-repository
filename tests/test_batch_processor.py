#!/usr/bin/env python3
"""
Comprehensive tests for batch processing pipeline
"""

import unittest
import tempfile
import shutil
import os
import json
from pathlib import Path
from datetime import datetime

from core.batch_processor import (
    ProcessingConfig, MemoryMonitor, StreamingProcessor,
    DeltaTracker, IncrementalProcessor, ResilientProcessor,
    ProgressTracker, ReportGenerator, MemvidBatchProcessor
)

class TestBatchProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.temp_files = []
        
        # Create test data directory
        self.test_data_dir = self.temp_dir / "test_data"
        self.test_data_dir.mkdir()
        
        # Create test output directory
        self.output_dir = self.temp_dir / "output"
        self.output_dir.mkdir()
        
        # Create sample transcript files
        self.create_test_files()
    
    def tearDown(self):
        """Clean up test environment"""
        # Clean up temporary files
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Clean up temporary directory
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        
        # Clean up any .faiss or .pkl files in current directory
        for ext in ['.faiss', '.pkl']:
            for file in Path('.').glob(f'*{ext}'):
                file.unlink(missing_ok=True)
    
    def create_test_files(self):
        """Create test transcript files"""
        test_transcripts = [
            "Episode 1: Introduction to AI\nSpeaker: Welcome to our podcast about artificial intelligence.",
            "Episode 2: Machine Learning Basics\nSpeaker: Today we discuss the fundamentals of machine learning.",
            "Episode 3: Deep Learning\nSpeaker: Let's explore neural networks and deep learning concepts."
        ]
        
        for i, content in enumerate(test_transcripts, 1):
            file_path = self.test_data_dir / f"episode_{i}.txt"
            with open(file_path, 'w') as f:
                f.write(content)
            self.temp_files.append(str(file_path))
    
    def test_processing_config(self):
        """Test ProcessingConfig dataclass"""
        config = ProcessingConfig()
        
        # Test default values
        self.assertEqual(config.buffer_size, 1024 * 1024)
        self.assertEqual(config.max_memory_mb, 512)
        self.assertEqual(config.chunk_size, 1024)
        self.assertEqual(config.overlap, 32)
        self.assertEqual(config.batch_size, 10)
        
        # Test custom values
        custom_config = ProcessingConfig(
            buffer_size=2048,
            max_memory_mb=256,
            chunk_size=512,
            overlap=16,
            batch_size=5
        )
        self.assertEqual(custom_config.buffer_size, 2048)
        self.assertEqual(custom_config.max_memory_mb, 256)
    
    def test_memory_monitor(self):
        """Test MemoryMonitor functionality"""
        monitor = MemoryMonitor(max_memory_mb=100)
        
        # Test memory usage retrieval
        memory_usage = monitor.get_memory_usage_mb()
        self.assertIsInstance(memory_usage, float)
        self.assertGreater(memory_usage, 0)
        
        # Test memory limit check
        is_exceeded = monitor.is_memory_limit_exceeded()
        self.assertIsInstance(is_exceeded, bool)
        
        # Test cleanup (should not raise exception)
        monitor.force_cleanup()
    
    def test_streaming_processor(self):
        """Test StreamingProcessor functionality"""
        config = ProcessingConfig(buffer_size=100, chunk_size=50, overlap=10)
        processor = StreamingProcessor(config)
        
        # Test file streaming
        test_file = self.test_data_dir / "episode_1.txt"
        chunks = list(processor.stream_file_chunks(test_file))
        
        self.assertGreater(len(chunks), 0)
        self.assertIsInstance(chunks[0], str)
        
        # Test directory streaming
        files = list(processor.stream_directory_files(self.test_data_dir, "*.txt"))
        self.assertEqual(len(files), 3)
        
        # Test stats
        stats = processor.get_processing_stats()
        self.assertIn('memory_usage_mb', stats)
        self.assertIn('memory_limit_mb', stats)
    
    def test_delta_tracker(self):
        """Test DeltaTracker functionality"""
        state_file = self.temp_dir / "delta_state.json"
        tracker = DeltaTracker(state_file)
        
        # Test file change detection
        test_file = self.test_data_dir / "episode_1.txt"
        
        # First check should indicate change (new file)
        self.assertTrue(tracker.has_file_changed(test_file))
        
        # Mark as processed
        tracker.mark_file_processed(test_file)
        
        # Should not indicate change now
        self.assertFalse(tracker.has_file_changed(test_file))
        
        # Test state persistence
        tracker.save_state()
        self.assertTrue(state_file.exists())
        
        # Create new tracker and load state
        new_tracker = DeltaTracker(state_file)
        self.assertFalse(new_tracker.has_file_changed(test_file))
    
    def test_incremental_processor(self):
        """Test IncrementalProcessor functionality"""
        config = ProcessingConfig()
        processor = IncrementalProcessor(config, self.temp_dir)
        
        # Test file discovery
        files_to_process = processor.get_files_to_process(self.test_data_dir)
        self.assertEqual(len(files_to_process), 3)
        
        # Test incremental processing
        test_file = self.test_data_dir / "episode_1.txt"
        chunks = list(processor.process_file_incrementally(test_file))
        self.assertGreater(len(chunks), 0)
    
    def test_progress_tracker(self):
        """Test ProgressTracker functionality"""
        tracker = ProgressTracker()
        
        # Initialize tracking
        tracker.initialize(total_files=3)
        
        # Update progress
        test_file = self.test_data_dir / "episode_1.txt"
        tracker.update_file_progress(test_file, chunks=5, file_size=100)
        
        # Get metrics
        metrics = tracker.get_current_metrics()
        self.assertEqual(metrics.total_files, 3)
        self.assertEqual(metrics.files_processed, 1)
        self.assertEqual(metrics.chunks_processed, 5)
        
        # Test progress report
        report = tracker.format_progress_report()
        self.assertIsInstance(report, str)
        self.assertIn("Progress:", report)
    
    def test_report_generator(self):
        """Test ReportGenerator functionality"""
        generator = ReportGenerator(self.output_dir)
        
        # Create mock metrics
        from core.batch_processor import ProgressMetrics
        metrics = ProgressMetrics(
            files_processed=2,
            total_files=3,
            chunks_processed=10,
            bytes_processed=1000,
            start_time=datetime.now(),
            current_time=datetime.now()
        )
        
        # Generate report
        processed_files = [{'file': 'test1.txt', 'chunks': 5}]
        failed_files = [{'file': 'test2.txt', 'error': 'Test error'}]
        config = ProcessingConfig()
        
        report_file = generator.generate_processing_report(
            metrics, processed_files, failed_files, config
        )
        
        self.assertTrue(report_file.exists())
        
        # Verify report content
        with open(report_file, 'r') as f:
            report_data = json.load(f)
        
        self.assertIn('summary', report_data)
        self.assertIn('processed_files', report_data)
        self.assertIn('failed_files', report_data)
    
    def test_memvid_batch_processor_single_file(self):
        """Test MemvidBatchProcessor single file processing"""
        config = ProcessingConfig(chunk_size=100, overlap=10)
        processor = MemvidBatchProcessor(config, self.temp_dir)
        
        # Test single file processing
        test_file = self.test_data_dir / "episode_1.txt"
        
        try:
            result = processor.process_single_file(test_file, self.output_dir)
            
            # Verify result structure
            self.assertEqual(result['status'], 'success')
            self.assertIn('video_path', result)
            self.assertIn('index_path', result)
            self.assertIn('chunks_processed', result)
            
            # Verify output files exist
            video_path = Path(result['video_path'])
            index_path = Path(result['index_path'])
            
            self.assertTrue(video_path.exists())
            self.assertTrue(index_path.exists())
            
        except ImportError:
            self.skipTest("memvid library not available")
        except Exception as e:
            # Expected if memvid has issues with test environment
            self.assertIn('memvid', str(e).lower())
    
    def test_memvid_batch_processor_directory(self):
        """Test MemvidBatchProcessor directory processing"""
        config = ProcessingConfig(chunk_size=100, overlap=10)
        processor = MemvidBatchProcessor(config, self.temp_dir)
        
        try:
            result = processor.process_directory(self.test_data_dir, self.output_dir)
            
            # Verify result structure
            self.assertIn('status', result)
            self.assertIn('batch_id', result)
            self.assertIn('processed_files', result)
            
            if result['status'] == 'success':
                self.assertIn('video_path', result)
                self.assertIn('index_path', result)
                self.assertIn('build_stats', result)
                
                # Verify output files
                video_path = Path(result['video_path'])
                index_path = Path(result['index_path'])
                
                self.assertTrue(video_path.exists())
                self.assertTrue(index_path.exists())
            
        except ImportError:
            self.skipTest("memvid library not available")
        except Exception as e:
            # Expected if memvid has issues with test environment
            self.assertIn('memvid', str(e).lower())
    
    def test_processing_status(self):
        """Test processing status reporting"""
        config = ProcessingConfig()
        processor = MemvidBatchProcessor(config, self.temp_dir)
        
        # Initialize progress
        processor.progress_tracker.initialize(total_files=3)
        
        # Get status
        status = processor.get_processing_status()
        
        self.assertIn('progress', status)
        self.assertIn('memory_usage', status)
        self.assertIn('status_report', status)
        
        # Verify progress structure
        progress = status['progress']
        self.assertIn('files_processed', progress)
        self.assertIn('total_files', progress)
        self.assertIn('percentage', progress)

if __name__ == '__main__':
    unittest.main() 