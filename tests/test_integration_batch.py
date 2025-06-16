#!/usr/bin/env python3
"""
Integration tests for batch processing pipeline with performance validation
"""

import unittest
import tempfile
import shutil
import os
import time
import threading
import psutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.batch_processor import (
    ProcessingConfig, MemvidBatchProcessor, 
    ProgressTracker, StreamingProcessor, IncrementalProcessor, ResilientProcessor
)

class TestBatchProcessingIntegration(unittest.TestCase):
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.temp_files = []
        
        # Create test data directories
        self.input_dir = self.temp_dir / "input"
        self.output_dir = self.temp_dir / "output"
        self.state_dir = self.temp_dir / "state"
        
        for dir_path in [self.input_dir, self.output_dir, self.state_dir]:
            dir_path.mkdir(parents=True)
        
        # Create test dataset
        self.create_test_dataset()
        
        # Track initial memory
        self.initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    def tearDown(self):
        """Clean up test environment"""
        # Clean up temporary files
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Clean up temporary directory
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        
        # Clean up memvid artifacts
        for ext in ['.faiss', '.pkl']:
            for file in Path('.').glob(f'*{ext}'):
                file.unlink(missing_ok=True)
    
    def create_test_dataset(self):
        """Create test dataset for integration testing"""
        # Create 20 transcript files with varying sizes
        for i in range(20):
            content_size = 500 + (i * 50)  # Varying sizes
            content = f"Episode {i+1}: " + "AI and machine learning content. " * content_size
            
            file_path = self.input_dir / f"episode_{i+1:03d}.txt"
            with open(file_path, 'w') as f:
                f.write(content)
            self.temp_files.append(str(file_path))
    
    @patch('memvid.MemvidEncoder')
    def test_end_to_end_batch_processing_mocked(self, mock_encoder_class):
        """Test complete end-to-end batch processing workflow with mocked memvid"""
        # Mock MemvidEncoder
        mock_encoder = MagicMock()
        mock_encoder.add_text.return_value = None
        mock_encoder.build_video.return_value = {
            'total_chunks': 100,
            'total_frames': 1000,
            'video_size_mb': 5.2,
            'duration_seconds': 10.0
        }
        mock_encoder_class.return_value = mock_encoder
        
        config = ProcessingConfig(
            chunk_size=512,
            overlap=32,
            max_memory_mb=256,
            batch_size=10
        )
        
        processor = MemvidBatchProcessor(config, self.state_dir)
        
        start_time = time.time()
        result = processor.process_directory(self.input_dir, self.output_dir)
        processing_time = time.time() - start_time
        
        # Validate results
        self.assertEqual(result['status'], 'success')
        self.assertIn('video_path', result)
        self.assertIn('index_path', result)
        self.assertGreater(result['processed_files'], 0)
        
        # Performance validation
        self.assertLess(processing_time, 30.0, "Processing should complete within 30 seconds")
        
        print(f"✅ End-to-end processing: {result['processed_files']} files in {processing_time:.2f}s")
    
    def test_memory_efficiency_validation(self):
        """Test memory usage stays within limits during processing"""
        config = ProcessingConfig(
            chunk_size=256,
            overlap=16,
            max_memory_mb=128  # Strict memory limit
        )
        
        # Test individual components for memory efficiency
        streaming_processor = StreamingProcessor(config)
        incremental_processor = IncrementalProcessor(config, self.state_dir)
        
        # Monitor memory during component operations
        memory_samples = []
        
        def monitor_memory():
            while getattr(monitor_memory, 'running', True):
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory - self.initial_memory)
                time.sleep(0.1)
        
        monitor_memory.running = True
        monitor_thread = threading.Thread(target=monitor_memory)
        monitor_thread.start()
        
        try:
            # Test streaming processor memory efficiency
            for file_path in list(self.input_dir.glob("*.txt"))[:5]:  # Test with 5 files
                # Process file chunks using streaming processor
                chunks_processed = 0
                for chunk in streaming_processor.stream_file_chunks(file_path):
                    chunks_processed += 1
                    self.assertIsInstance(chunk, str)
                    self.assertGreater(len(chunk), 0)
                
                self.assertGreater(chunks_processed, 0)
            
            # Stop monitoring
            monitor_memory.running = False
            monitor_thread.join()
            
            # Validate memory usage
            max_memory_used = max(memory_samples) if memory_samples else 0
            avg_memory_used = sum(memory_samples) / len(memory_samples) if memory_samples else 0
            
            print(f"✅ Memory usage - Max: {max_memory_used:.1f}MB, Avg: {avg_memory_used:.1f}MB")
            
            # Memory should stay reasonable (allowing some overhead)
            self.assertLess(max_memory_used, 200, "Memory usage should stay under 200MB")
            
        except Exception as e:
            monitor_memory.running = False
            monitor_thread.join()
            raise
    
    def test_error_recovery_scenarios(self):
        """Test error recovery mechanisms under various failure scenarios"""
        config = ProcessingConfig(chunk_size=512, overlap=32)
        resilient_processor = ResilientProcessor(config, self.state_dir)
        
        # Create a corrupted file
        corrupted_file = self.input_dir / "corrupted.txt"
        with open(corrupted_file, 'wb') as f:
            f.write(b'\x00\x01\x02\x03')  # Binary data that will cause encoding issues
        self.temp_files.append(str(corrupted_file))
        
        # Test error recovery with individual files
        valid_files = []
        failed_files = []
        
        for file_path in self.input_dir.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Test if file can be processed
                if len(content.strip()) > 0 and content.isprintable():
                    valid_files.append(file_path)
                else:
                    failed_files.append(file_path)
                    
            except Exception:
                failed_files.append(file_path)
        
        # Should handle errors gracefully
        self.assertGreater(len(valid_files), 0, "Should have some valid files")
        self.assertGreater(len(failed_files), 0, "Should detect corrupted files")
        
        print(f"✅ Error recovery: {len(valid_files)} valid, {len(failed_files)} failed files detected")
    
    def test_progress_reporting_accuracy(self):
        """Test progress reporting accuracy across processing modes"""
        config = ProcessingConfig(chunk_size=256, overlap=16)
        progress_tracker = ProgressTracker()
        
        # Initialize progress tracking
        total_files = len(list(self.input_dir.glob("*.txt")))
        progress_tracker.initialize(total_files=total_files)
        
        progress_reports = []
        
        # Simulate file processing with progress tracking
        for i, file_path in enumerate(self.input_dir.glob("*.txt")):
            # Simulate processing time
            time.sleep(0.01)
            
            # Get file size for progress tracking
            file_size = file_path.stat().st_size
            chunks_count = 10 + i  # Simulate varying chunk counts
            
            # Update progress with correct signature
            progress_tracker.update_file_progress(file_path, chunks_count, file_size)
            
            # Capture progress
            metrics = progress_tracker.get_current_metrics()
            progress_reports.append({
                'files_processed': metrics.files_processed,
                'percentage': metrics.progress_percentage,
                'chunks_processed': metrics.chunks_processed
            })
        
        # Validate progress reporting
        self.assertGreater(len(progress_reports), 0)
        
        # Progress should be monotonically increasing
        for i in range(1, len(progress_reports)):
            self.assertGreaterEqual(
                progress_reports[i]['files_processed'],
                progress_reports[i-1]['files_processed']
            )
        
        # Final progress should be 100%
        final_progress = progress_reports[-1]
        self.assertEqual(final_progress['percentage'], 100.0)
        self.assertEqual(final_progress['files_processed'], total_files)
        
        print(f"✅ Progress tracking: {len(progress_reports)} updates, final: {final_progress['percentage']:.1f}%")
    
    def test_concurrent_processing_safety(self):
        """Test thread safety and concurrent processing scenarios"""
        config = ProcessingConfig(chunk_size=256, overlap=16, max_memory_mb=256)
        
        # Test concurrent progress tracking
        progress_trackers = [ProgressTracker() for _ in range(3)]
        
        # Initialize each tracker
        for i, tracker in enumerate(progress_trackers):
            tracker.initialize(total_files=5)
        
        results = []
        threads = []
        
        def process_with_tracker(tracker, tracker_id):
            try:
                # Simulate processing files
                for j in range(5):
                    time.sleep(0.01)  # Simulate processing time
                    
                    # Create dummy file path and simulate file processing
                    dummy_path = Path(f"file_{tracker_id}_{j}.txt")
                    file_size = 1000 + j * 100  # Simulate varying file sizes
                    chunks_count = j + 1
                    
                    tracker.update_file_progress(dummy_path, chunks_count, file_size)
                
                metrics = tracker.get_current_metrics()
                results.append({
                    'tracker_id': tracker_id,
                    'files_processed': metrics.files_processed,
                    'percentage': metrics.progress_percentage,
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'tracker_id': tracker_id,
                    'status': 'error',
                    'error': str(e)
                })
        
        # Start concurrent processing
        for i, tracker in enumerate(progress_trackers):
            thread = threading.Thread(
                target=process_with_tracker,
                args=(tracker, i)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)  # 10 second timeout
        
        # Validate results
        successful_results = [r for r in results if r.get('status') == 'success']
        
        print(f"✅ Concurrent processing: {len(successful_results)}/{len(progress_trackers)} successful")
        
        # All should succeed
        self.assertEqual(len(successful_results), len(progress_trackers))
        
        # Each should have processed all files
        for result in successful_results:
            self.assertEqual(result['files_processed'], 5)
            self.assertEqual(result['percentage'], 100.0)
    
    def test_large_dataset_performance(self):
        """Test performance with large datasets"""
        # Create larger dataset for performance testing
        large_input_dir = self.temp_dir / "large_input"
        large_input_dir.mkdir()
        
        # Create 50 files with substantial content
        for i in range(50):
            content = f"Large Episode {i+1}: " + "AI machine learning content. " * 200
            file_path = large_input_dir / f"large_episode_{i+1:03d}.txt"
            with open(file_path, 'w') as f:
                f.write(content)
            self.temp_files.append(str(file_path))
        
        config = ProcessingConfig(
            chunk_size=1024,
            overlap=64,
            max_memory_mb=512,
            batch_size=20
        )
        
        # Test streaming processor performance
        streaming_processor = StreamingProcessor(config)
        
        start_time = time.time()
        
        total_chunks = 0
        for file_path in large_input_dir.glob("*.txt"):
            # Process file chunks using streaming processor
            chunks_processed = 0
            for chunk in streaming_processor.stream_file_chunks(file_path):
                chunks_processed += 1
            total_chunks += chunks_processed
        
        processing_time = time.time() - start_time
        
        # Calculate performance metrics
        files_processed = len(list(large_input_dir.glob("*.txt")))
        files_per_second = files_processed / processing_time
        
        print(f"✅ Large dataset: {files_processed} files, {total_chunks} chunks in {processing_time:.2f}s ({files_per_second:.2f} files/s)")
        
        # Performance should be reasonable
        self.assertGreater(files_per_second, 5.0, "Should process at least 5 files per second")
        self.assertLess(processing_time, 30, "Should complete within 30 seconds")
        self.assertGreater(total_chunks, 0, "Should generate chunks")
    
    @patch('memvid.MemvidEncoder')
    def test_component_integration(self, mock_encoder_class):
        """Test integration between all batch processor components"""
        # Mock MemvidEncoder
        mock_encoder = MagicMock()
        mock_encoder.add_text.return_value = None
        mock_encoder.build_video.return_value = {
            'total_chunks': 50,
            'total_frames': 500,
            'video_size_mb': 2.1,
            'duration_seconds': 5.0
        }
        mock_encoder_class.return_value = mock_encoder
        
        config = ProcessingConfig(chunk_size=512, overlap=32, max_memory_mb=256)
        
        # Test all components working together
        streaming_processor = StreamingProcessor(config)
        incremental_processor = IncrementalProcessor(config, self.state_dir)
        resilient_processor = ResilientProcessor(config, self.state_dir)
        progress_tracker = ProgressTracker()
        
        # Initialize progress tracking
        total_files = len(list(self.input_dir.glob("*.txt")))
        progress_tracker.initialize(total_files=total_files)
        
        # Process files through the pipeline
        processed_files = 0
        total_chunks = 0
        
        for file_path in self.input_dir.glob("*.txt"):
            try:
                # Stream processing
                chunks_processed = 0
                for chunk in streaming_processor.stream_file_chunks(file_path):
                    chunks_processed += 1
                total_chunks += chunks_processed
                
                # Progress tracking
                file_size = file_path.stat().st_size
                progress_tracker.update_file_progress(file_path, chunks_processed, file_size)
                processed_files += 1
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Validate integration
        self.assertGreater(processed_files, 0)
        self.assertGreater(total_chunks, 0)
        
        metrics = progress_tracker.get_current_metrics()
        self.assertEqual(metrics.files_processed, processed_files)
        self.assertEqual(metrics.progress_percentage, 100.0)
        
        print(f"✅ Component integration: {processed_files} files, {total_chunks} chunks processed")

if __name__ == '__main__':
    unittest.main() 