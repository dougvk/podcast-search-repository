import os
import gc
import psutil
import json
import hashlib
import time
import traceback
from pathlib import Path
from typing import Iterator, Dict, Any, Optional, List, Set, Callable
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for batch processing"""
    buffer_size: int = 1024 * 1024  # 1MB buffer
    max_memory_mb: int = 512  # Max memory usage
    chunk_size: int = 1024  # Text chunk size
    overlap: int = 32  # Chunk overlap
    batch_size: int = 10  # Episodes per batch

class MemoryMonitor:
    """Monitor and control memory usage"""
    
    def __init__(self, max_memory_mb: int):
        self.max_memory_mb = max_memory_mb
        self.process = psutil.Process()
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def is_memory_limit_exceeded(self) -> bool:
        """Check if memory limit is exceeded"""
        return self.get_memory_usage_mb() > self.max_memory_mb
    
    def force_cleanup(self):
        """Force garbage collection and memory cleanup"""
        gc.collect()
        logger.debug(f"Memory after cleanup: {self.get_memory_usage_mb():.1f}MB")

class StreamingProcessor:
    """Memory-efficient streaming processor for large datasets"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.memory_monitor = MemoryMonitor(config.max_memory_mb)
    
    @contextmanager
    def memory_managed_processing(self):
        """Context manager for memory-managed processing"""
        initial_memory = self.memory_monitor.get_memory_usage_mb()
        logger.debug(f"Starting processing with {initial_memory:.1f}MB memory")
        
        try:
            yield
        finally:
            self.memory_monitor.force_cleanup()
            final_memory = self.memory_monitor.get_memory_usage_mb()
            logger.debug(f"Finished processing with {final_memory:.1f}MB memory")
    
    def stream_file_chunks(self, file_path: Path) -> Iterator[str]:
        """Stream file content in chunks to minimize memory usage"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            buffer = ""
            while True:
                chunk = f.read(self.config.buffer_size)
                if not chunk:
                    if buffer:
                        yield buffer
                    break
                
                buffer += chunk
                
                # Yield complete chunks and keep remainder
                while len(buffer) >= self.config.chunk_size:
                    yield buffer[:self.config.chunk_size]
                    buffer = buffer[self.config.chunk_size - self.config.overlap:]
                
                # Check memory usage and cleanup if needed
                if self.memory_monitor.is_memory_limit_exceeded():
                    self.memory_monitor.force_cleanup()
    
    def stream_directory_files(self, directory: Path, pattern: str = "*.txt") -> Iterator[Path]:
        """Stream files from directory without loading all into memory"""
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                yield file_path
                
                # Periodic memory check
                if self.memory_monitor.is_memory_limit_exceeded():
                    logger.warning(f"Memory limit exceeded: {self.memory_monitor.get_memory_usage_mb():.1f}MB")
                    self.memory_monitor.force_cleanup()
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return {
            'memory_usage_mb': self.memory_monitor.get_memory_usage_mb(),
            'memory_limit_mb': self.config.max_memory_mb,
            'memory_utilization': self.memory_monitor.get_memory_usage_mb() / self.config.max_memory_mb,
            'buffer_size': self.config.buffer_size,
            'chunk_size': self.config.chunk_size
        }

@dataclass
class FileMetadata:
    """Metadata for tracking file changes"""
    path: str
    size: int
    modified_time: float
    hash: str
    processed_time: Optional[datetime] = None

class DeltaTracker:
    """Track file changes for incremental processing"""
    
    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.file_states: Dict[str, FileMetadata] = {}
        self.load_state()
    
    def load_state(self):
        """Load previous processing state"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.file_states = {
                        path: FileMetadata(**meta) for path, meta in data.items()
                    }
                logger.info(f"Loaded state for {len(self.file_states)} files")
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")
    
    def save_state(self):
        """Save current processing state"""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                data = {path: asdict(meta) for path, meta in self.file_states.items()}
                json.dump(data, f, indent=2, default=str)
            logger.debug(f"Saved state for {len(self.file_states)} files")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def get_file_hash(self, file_path: Path) -> str:
        """Calculate file hash for change detection"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def has_file_changed(self, file_path: Path) -> bool:
        """Check if file has changed since last processing"""
        path_str = str(file_path)
        
        if path_str not in self.file_states:
            return True  # New file
        
        try:
            stat = file_path.stat()
            current_hash = self.get_file_hash(file_path)
            stored = self.file_states[path_str]
            
            return (stat.st_size != stored.size or 
                   stat.st_mtime != stored.modified_time or 
                   current_hash != stored.hash)
        except Exception:
            return True  # Assume changed if can't check
    
    def mark_file_processed(self, file_path: Path):
        """Mark file as processed with current metadata"""
        try:
            stat = file_path.stat()
            file_hash = self.get_file_hash(file_path)
            
            self.file_states[str(file_path)] = FileMetadata(
                path=str(file_path),
                size=stat.st_size,
                modified_time=stat.st_mtime,
                hash=file_hash,
                processed_time=datetime.now()
            )
        except Exception as e:
            logger.error(f"Failed to mark file processed: {e}")
    
    def get_changed_files(self, file_paths: List[Path]) -> List[Path]:
        """Get list of files that have changed"""
        changed = []
        for file_path in file_paths:
            if self.has_file_changed(file_path):
                changed.append(file_path)
        return changed
    
    def cleanup_deleted_files(self, current_files: Set[Path]):
        """Remove state for files that no longer exist"""
        current_paths = {str(p) for p in current_files}
        to_remove = [path for path in self.file_states.keys() if path not in current_paths]
        
        for path in to_remove:
            del self.file_states[path]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} deleted files from state")

class IncrementalProcessor:
    """Incremental processing with delta tracking"""
    
    def __init__(self, config: ProcessingConfig, state_dir: Path):
        self.config = config
        self.state_dir = state_dir
        self.delta_tracker = DeltaTracker(state_dir / "processing_state.json")
        self.streaming_processor = StreamingProcessor(config)
    
    def get_files_to_process(self, directory: Path, pattern: str = "*.txt") -> List[Path]:
        """Get files that need processing (new or changed)"""
        all_files = list(directory.glob(pattern))
        
        # Cleanup deleted files from state
        self.delta_tracker.cleanup_deleted_files(set(all_files))
        
        # Get changed files
        changed_files = self.delta_tracker.get_changed_files(all_files)
        
        logger.info(f"Found {len(all_files)} total files, {len(changed_files)} need processing")
        return changed_files
    
    def process_file_incrementally(self, file_path: Path) -> Iterator[str]:
        """Process file and mark as processed"""
        try:
            # Stream file chunks
            for chunk in self.streaming_processor.stream_file_chunks(file_path):
                yield chunk
            
            # Mark as processed
            self.delta_tracker.mark_file_processed(file_path)
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            raise
    
    def save_checkpoint(self):
        """Save processing checkpoint"""
        self.delta_tracker.save_state()

class ProcessingStatus(Enum):
    """Processing status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

@dataclass
class ProcessingCheckpoint:
    """Checkpoint data for recovery"""
    batch_id: str
    files_processed: int
    total_files: int
    current_file: Optional[str]
    status: ProcessingStatus
    error_count: int
    last_error: Optional[str]
    timestamp: datetime
    retry_count: int = 0

class ErrorRecoveryManager:
    """Manage error recovery and retry logic"""
    
    def __init__(self, checkpoint_file: Path, max_retries: int = 3):
        self.checkpoint_file = checkpoint_file
        self.max_retries = max_retries
        self.current_checkpoint: Optional[ProcessingCheckpoint] = None
        self.load_checkpoint()
    
    def load_checkpoint(self):
        """Load existing checkpoint"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    data['status'] = ProcessingStatus(data['status'])
                    data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                    self.current_checkpoint = ProcessingCheckpoint(**data)
                logger.info(f"Loaded checkpoint: {self.current_checkpoint.batch_id}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
    
    def save_checkpoint(self, checkpoint: ProcessingCheckpoint):
        """Save checkpoint to disk"""
        try:
            self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            data = asdict(checkpoint)
            data['status'] = checkpoint.status.value
            data['timestamp'] = checkpoint.timestamp.isoformat()
            
            with open(self.checkpoint_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.current_checkpoint = checkpoint
            logger.debug(f"Saved checkpoint: {checkpoint.batch_id}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def should_retry(self, error: Exception) -> bool:
        """Determine if error should trigger retry"""
        if not self.current_checkpoint:
            return True
        
        return self.current_checkpoint.retry_count < self.max_retries
    
    def record_error(self, error: Exception):
        """Record error and update retry count"""
        if self.current_checkpoint:
            self.current_checkpoint.error_count += 1
            self.current_checkpoint.last_error = str(error)
            self.current_checkpoint.retry_count += 1
            self.current_checkpoint.status = ProcessingStatus.FAILED
            self.save_checkpoint(self.current_checkpoint)
    
    def reset_retry_count(self):
        """Reset retry count after successful operation"""
        if self.current_checkpoint:
            self.current_checkpoint.retry_count = 0

class ResilientProcessor:
    """Processor with error recovery and resume capabilities"""
    
    def __init__(self, config: ProcessingConfig, state_dir: Path):
        self.config = config
        self.state_dir = state_dir
        self.incremental_processor = IncrementalProcessor(config, state_dir)
        self.error_manager = ErrorRecoveryManager(state_dir / "checkpoint.json")
    
    def exponential_backoff(self, attempt: int, base_delay: float = 1.0) -> float:
        """Calculate exponential backoff delay"""
        return min(base_delay * (2 ** attempt), 60.0)  # Max 60 seconds
    
    def retry_with_backoff(self, func: Callable, *args, **kwargs):
        """Execute function with exponential backoff retry"""
        for attempt in range(self.error_manager.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                self.error_manager.reset_retry_count()
                return result
            except Exception as e:
                if attempt == self.error_manager.max_retries:
                    self.error_manager.record_error(e)
                    raise
                
                delay = self.exponential_backoff(attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s")
                time.sleep(delay)
    
    def process_with_recovery(self, directory: Path, batch_id: str) -> Dict[str, Any]:
        """Process directory with error recovery"""
        files_to_process = self.incremental_processor.get_files_to_process(directory)
        
        # Create or resume checkpoint
        checkpoint = ProcessingCheckpoint(
            batch_id=batch_id,
            files_processed=0,
            total_files=len(files_to_process),
            current_file=None,
            status=ProcessingStatus.RUNNING,
            error_count=0,
            last_error=None,
            timestamp=datetime.now()
        )
        
        # Resume from existing checkpoint if available
        if (self.error_manager.current_checkpoint and 
            self.error_manager.current_checkpoint.batch_id == batch_id):
            checkpoint = self.error_manager.current_checkpoint
            checkpoint.status = ProcessingStatus.RUNNING
            logger.info(f"Resuming from checkpoint: {checkpoint.files_processed}/{checkpoint.total_files}")
        
        self.error_manager.save_checkpoint(checkpoint)
        
        processed_files = []
        failed_files = []
        
        try:
            for i, file_path in enumerate(files_to_process[checkpoint.files_processed:], 
                                        checkpoint.files_processed):
                checkpoint.current_file = str(file_path)
                checkpoint.files_processed = i
                self.error_manager.save_checkpoint(checkpoint)
                
                try:
                    # Process file with retry
                    chunks = list(self.retry_with_backoff(
                        self.incremental_processor.process_file_incrementally,
                        file_path
                    ))
                    
                    processed_files.append({
                        'file': str(file_path),
                        'chunks': len(chunks),
                        'status': 'success'
                    })
                    
                    logger.info(f"Processed {file_path.name}: {len(chunks)} chunks")
                    
                except Exception as e:
                    failed_files.append({
                        'file': str(file_path),
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    })
                    logger.error(f"Failed to process {file_path}: {e}")
                    
                    # Continue with next file instead of stopping
                    continue
                
                # Save checkpoint every 10 files
                if (i + 1) % 10 == 0:
                    self.incremental_processor.save_checkpoint()
            
            # Mark as completed
            checkpoint.status = ProcessingStatus.COMPLETED
            checkpoint.files_processed = len(files_to_process)
            self.error_manager.save_checkpoint(checkpoint)
            
            return {
                'status': 'completed',
                'processed_files': processed_files,
                'failed_files': failed_files,
                'total_files': len(files_to_process),
                'success_count': len(processed_files),
                'failure_count': len(failed_files)
            }
            
        except Exception as e:
            checkpoint.status = ProcessingStatus.FAILED
            checkpoint.last_error = str(e)
            self.error_manager.save_checkpoint(checkpoint)
            raise
    
    def can_resume(self, batch_id: str) -> bool:
        """Check if processing can be resumed"""
        return (self.error_manager.current_checkpoint and 
                self.error_manager.current_checkpoint.batch_id == batch_id and
                self.error_manager.current_checkpoint.status in [ProcessingStatus.FAILED, ProcessingStatus.RUNNING])
    
    def get_resume_info(self) -> Optional[Dict[str, Any]]:
        """Get information about resumable processing"""
        if not self.error_manager.current_checkpoint:
            return None
        
        cp = self.error_manager.current_checkpoint
        return {
            'batch_id': cp.batch_id,
            'progress': f"{cp.files_processed}/{cp.total_files}",
            'status': cp.status.value,
            'last_error': cp.last_error,
                         'can_resume': cp.status in [ProcessingStatus.FAILED, ProcessingStatus.RUNNING]
         }

@dataclass
class ProgressMetrics:
    """Progress tracking metrics"""
    files_processed: int
    total_files: int
    chunks_processed: int
    bytes_processed: int
    start_time: datetime
    current_time: datetime
    estimated_completion: Optional[datetime] = None
    processing_rate: float = 0.0  # files per second
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage"""
        if self.total_files == 0:
            return 0.0
        return (self.files_processed / self.total_files) * 100
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        return (self.current_time - self.start_time).total_seconds()
    
    @property
    def estimated_remaining(self) -> Optional[float]:
        """Estimate remaining time in seconds"""
        if self.processing_rate <= 0 or self.files_processed == 0:
            return None
        
        remaining_files = self.total_files - self.files_processed
        return remaining_files / self.processing_rate

class ProgressTracker:
    """Track and report processing progress"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.files_processed = 0
        self.total_files = 0
        self.chunks_processed = 0
        self.bytes_processed = 0
        self.file_times: List[float] = []
        self.last_update = self.start_time
    
    def initialize(self, total_files: int):
        """Initialize progress tracking"""
        self.total_files = total_files
        self.start_time = datetime.now()
        self.last_update = self.start_time
        logger.info(f"Starting progress tracking for {total_files} files")
    
    def update_file_progress(self, file_path: Path, chunks: int, file_size: int):
        """Update progress for completed file"""
        current_time = datetime.now()
        file_duration = (current_time - self.last_update).total_seconds()
        
        self.files_processed += 1
        self.chunks_processed += chunks
        self.bytes_processed += file_size
        self.file_times.append(file_duration)
        self.last_update = current_time
        
        # Keep only recent file times for rate calculation
        if len(self.file_times) > 50:
            self.file_times = self.file_times[-50:]
    
    def get_current_metrics(self) -> ProgressMetrics:
        """Get current progress metrics"""
        current_time = datetime.now()
        
        # Calculate processing rate
        if self.file_times and self.files_processed > 0:
            avg_time_per_file = sum(self.file_times) / len(self.file_times)
            processing_rate = 1.0 / avg_time_per_file if avg_time_per_file > 0 else 0.0
        else:
            processing_rate = 0.0
        
        metrics = ProgressMetrics(
            files_processed=self.files_processed,
            total_files=self.total_files,
            chunks_processed=self.chunks_processed,
            bytes_processed=self.bytes_processed,
            start_time=self.start_time,
            current_time=current_time,
            processing_rate=processing_rate
        )
        
        # Calculate estimated completion
        if metrics.estimated_remaining:
            metrics.estimated_completion = current_time + timedelta(seconds=metrics.estimated_remaining)
        
        return metrics
    
    def format_progress_report(self) -> str:
        """Format progress as human-readable report"""
        metrics = self.get_current_metrics()
        
        report = []
        report.append(f"Progress: {metrics.files_processed}/{metrics.total_files} files ({metrics.progress_percentage:.1f}%)")
        report.append(f"Chunks processed: {metrics.chunks_processed:,}")
        report.append(f"Data processed: {self._format_bytes(metrics.bytes_processed)}")
        report.append(f"Elapsed time: {self._format_duration(metrics.elapsed_time)}")
        report.append(f"Processing rate: {metrics.processing_rate:.2f} files/sec")
        
        if metrics.estimated_remaining:
            report.append(f"Estimated remaining: {self._format_duration(metrics.estimated_remaining)}")
            if metrics.estimated_completion:
                report.append(f"Estimated completion: {metrics.estimated_completion.strftime('%H:%M:%S')}")
        
        return "\n".join(report)
    
    def _format_bytes(self, bytes_count: int) -> str:
        """Format bytes in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_count < 1024:
                return f"{bytes_count:.1f} {unit}"
            bytes_count /= 1024
        return f"{bytes_count:.1f} TB"
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human readable format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"

class ReportGenerator:
    """Generate processing reports"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_processing_report(self, 
                                 metrics: ProgressMetrics,
                                 processed_files: List[Dict],
                                 failed_files: List[Dict],
                                 config: ProcessingConfig) -> Path:
        """Generate comprehensive processing report"""
        
        report_data = {
            'summary': {
                'total_files': metrics.total_files,
                'processed_files': metrics.files_processed,
                'failed_files': len(failed_files),
                'success_rate': (metrics.files_processed / metrics.total_files * 100) if metrics.total_files > 0 else 0,
                'chunks_processed': metrics.chunks_processed,
                'bytes_processed': metrics.bytes_processed,
                'processing_time': metrics.elapsed_time,
                'processing_rate': metrics.processing_rate
            },
            'configuration': asdict(config),
            'processed_files': processed_files,
            'failed_files': failed_files,
            'timestamps': {
                'start_time': metrics.start_time.isoformat(),
                'end_time': metrics.current_time.isoformat(),
                'duration_seconds': metrics.elapsed_time
            }
        }
        
        # Save JSON report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"processing_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Generate human-readable summary
        summary_file = self.output_dir / f"processing_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("BATCH PROCESSING REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total files: {metrics.total_files}\n")
            f.write(f"Successfully processed: {metrics.files_processed}\n")
            f.write(f"Failed: {len(failed_files)}\n")
            f.write(f"Success rate: {report_data['summary']['success_rate']:.1f}%\n\n")
            f.write(f"Chunks generated: {metrics.chunks_processed:,}\n")
            f.write(f"Data processed: {self._format_bytes(metrics.bytes_processed)}\n")
            f.write(f"Processing time: {self._format_duration(metrics.elapsed_time)}\n")
            f.write(f"Average rate: {metrics.processing_rate:.2f} files/sec\n\n")
            
            if failed_files:
                f.write("FAILED FILES:\n")
                f.write("-" * 20 + "\n")
                for failed in failed_files:
                    f.write(f"- {failed['file']}: {failed['error']}\n")
        
        logger.info(f"Generated processing report: {report_file}")
        return report_file
    
    def _format_bytes(self, bytes_count: int) -> str:
        """Format bytes in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_count < 1024:
                return f"{bytes_count:.1f} {unit}"
            bytes_count /= 1024
        return f"{bytes_count:.1f} TB"
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human readable format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"

class MemvidBatchProcessor:
    """Optimized batch processor using memvid library"""
    
    def __init__(self, config: ProcessingConfig, state_dir: Path):
        self.config = config
        self.state_dir = state_dir
        self.resilient_processor = ResilientProcessor(config, state_dir)
        self.progress_tracker = ProgressTracker()
        self.report_generator = ReportGenerator(state_dir / "reports")
    
    def process_directory(self, input_dir: Path, output_dir: Path, batch_id: str = None) -> Dict[str, Any]:
        """Process entire directory of transcript files into memvid dataset"""
        from memvid import MemvidEncoder
        
        if not batch_id:
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting memvid batch processing: {batch_id}")
        
        # Get files to process
        files_to_process = list(input_dir.glob("*.txt"))
        if not files_to_process:
            return {'status': 'no_files', 'message': 'No transcript files found'}
        
        # Initialize progress tracking
        self.progress_tracker.initialize(len(files_to_process))
        
        # Create optimized encoder config
        encoder_config = {
            "chunking": {
                "chunk_size": self.config.chunk_size,
                "overlap": self.config.overlap
            }
        }
        
        try:
            # Initialize encoder with config
            encoder = MemvidEncoder(config=encoder_config)
            
            # Process files with recovery
            processed_files = []
            failed_files = []
            
            for file_path in files_to_process:
                try:
                    # Read and add file content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Add to encoder with chunking
                    encoder.add_text(content, self.config.chunk_size, self.config.overlap)
                    
                    # Update progress
                    file_size = file_path.stat().st_size
                    chunks_added = len(content) // self.config.chunk_size + 1
                    self.progress_tracker.update_file_progress(file_path, chunks_added, file_size)
                    
                    processed_files.append({
                        'file': str(file_path),
                        'chunks': chunks_added,
                        'size': file_size
                    })
                    
                    logger.debug(f"Added {file_path.name} to encoder")
                    
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    failed_files.append({'file': str(file_path), 'error': str(e)})
            
            # Build video dataset
            output_dir.mkdir(parents=True, exist_ok=True)
            video_path = output_dir / f"{batch_id}.mp4"
            index_path = output_dir / f"{batch_id}_index.json"
            
            logger.info(f"Building memvid dataset: {video_path}")
            build_stats = encoder.build_video(str(video_path), str(index_path), show_progress=False)
            
            # Generate final report
            final_metrics = self.progress_tracker.get_current_metrics()
            report_file = self.report_generator.generate_processing_report(
                final_metrics, processed_files, failed_files, self.config
            )
            
            result = {
                'status': 'success',
                'batch_id': batch_id,
                'video_path': str(video_path),
                'index_path': str(index_path),
                'processed_files': len(processed_files),
                'failed_files': len(failed_files),
                'build_stats': build_stats,
                'report_file': str(report_file)
            }
            
            logger.info(f"Batch processing completed: {len(processed_files)} files processed")
            return result
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise
    
    def process_single_file(self, file_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Process single transcript file with memvid"""
        from memvid import MemvidEncoder
        
        logger.info(f"Processing single file: {file_path}")
        
        try:
            # Create encoder config
            encoder_config = {
                "chunking": {
                    "chunk_size": self.config.chunk_size,
                    "overlap": self.config.overlap
                }
            }
            encoder = MemvidEncoder(config=encoder_config)
            
            # Read and process file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add to encoder
            encoder.add_text(content, self.config.chunk_size, self.config.overlap)
            
            # Build output files
            output_dir.mkdir(parents=True, exist_ok=True)
            base_name = file_path.stem
            video_path = output_dir / f"{base_name}.mp4"
            index_path = output_dir / f"{base_name}_index.json"
            
            # Build video
            build_stats = encoder.build_video(str(video_path), str(index_path), show_progress=False)
            
            # Update progress
            file_size = file_path.stat().st_size
            chunks_added = len(content) // self.config.chunk_size + 1
            self.progress_tracker.update_file_progress(file_path, chunks_added, file_size)
            
            result = {
                'file': str(file_path),
                'video_path': str(video_path),
                'index_path': str(index_path),
                'chunks_processed': chunks_added,
                'build_stats': build_stats,
                'status': 'success'
            }
            
            logger.info(f"Successfully processed {file_path.name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            raise
    
    def optimize_dataset(self, encoder, total_files: int) -> Dict[str, Any]:
        """Optimize memvid dataset for large collections"""
        logger.info(f"Optimizing dataset for {total_files} files")
        
        # Get encoder stats
        stats = encoder.get_stats()
        
        # Optimize based on dataset size
        optimizations = {
            'chunk_optimization': False,
            'index_optimization': False,
            'compression_optimization': False
        }
        
        # Large dataset optimizations
        if total_files > 100:
            # Use larger chunks for better compression
            if stats['avg_chunk_size'] < 2048:
                optimizations['chunk_optimization'] = True
                logger.info("Recommending larger chunk size for better compression")
        
        if stats['total_chunks'] > 10000:
            # Use IVF index for faster search
            optimizations['index_optimization'] = True
            logger.info("Recommending IVF index for large chunk count")
        
        if stats['total_characters'] > 10_000_000:  # 10MB+ text
            # Use H.265 codec for better compression
            optimizations['compression_optimization'] = True
            logger.info("Recommending H.265 codec for large dataset")
        
        return {
            'dataset_stats': stats,
            'optimizations': optimizations,
            'recommendations': self._get_optimization_recommendations(stats, total_files)
        }
    
    def _get_optimization_recommendations(self, stats: Dict, total_files: int) -> List[str]:
        """Get specific optimization recommendations"""
        recommendations = []
        
        if stats['total_chunks'] > 50000:
            recommendations.append("Consider using IVF index type for faster search")
        
        if stats['avg_chunk_size'] < 1024 and total_files > 50:
            recommendations.append("Increase chunk_size to 2048 for better compression")
        
        if stats['total_characters'] > 5_000_000:
            recommendations.append("Use H.265 codec for better compression ratio")
        
        if total_files > 1000:
            recommendations.append("Process in smaller batches to manage memory usage")
        
        return recommendations
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status"""
        metrics = self.progress_tracker.get_current_metrics()
        
        return {
            'progress': {
                'files_processed': metrics.files_processed,
                'total_files': metrics.total_files,
                'percentage': metrics.progress_percentage,
                'chunks_processed': metrics.chunks_processed,
                'processing_rate': metrics.processing_rate
            },
            'memory_usage': self.resilient_processor.incremental_processor.streaming_processor.memory_monitor.get_memory_usage_mb(),
            'status_report': self.progress_tracker.format_progress_report()
        }