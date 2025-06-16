"""
High-Performance Monitoring and Profiling System

Minimal code for maximum performance insights. Follows memvid patterns
with real-time metrics, bottleneck detection, and automatic profiling.
"""

import time
import threading
import psutil
import os
import logging
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from functools import wraps
import json

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_bytes_sent: float
    network_bytes_recv: float
    timestamp: float

class PerformanceProfiler:
    """Lightweight performance profiler with automatic bottleneck detection"""
    
    def __init__(self, max_samples: int = 1000):
        self._lock = threading.RLock()
        self._max_samples = max_samples
        self._metrics = deque(maxlen=max_samples)
        self._function_stats = defaultdict(lambda: {"calls": 0, "total_time": 0.0, "avg_time": 0.0})
        self._endpoint_stats = defaultdict(lambda: {"requests": 0, "total_time": 0.0, "errors": 0})
        self._bottlenecks = []
        self._monitoring = False
        self._monitor_thread = None
    
    def start_monitoring(self, interval: float = 1.0):
        """Start continuous system monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,), daemon=True)
        self._monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop"""
        last_io = psutil.disk_io_counters()
        last_net = psutil.net_io_counters()
        
        while self._monitoring:
            try:
                # System metrics
                cpu = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                
                # IO metrics
                current_io = psutil.disk_io_counters()
                disk_read = (current_io.read_bytes - last_io.read_bytes) / (1024*1024) if last_io else 0
                disk_write = (current_io.write_bytes - last_io.write_bytes) / (1024*1024) if last_io else 0
                last_io = current_io
                
                # Network metrics
                current_net = psutil.net_io_counters()
                net_sent = (current_net.bytes_sent - last_net.bytes_sent) if last_net else 0
                net_recv = (current_net.bytes_recv - last_net.bytes_recv) if last_net else 0
                last_net = current_net
                
                metrics = PerformanceMetrics(
                    cpu_percent=cpu,
                    memory_percent=memory.percent,
                    memory_mb=memory.used / (1024*1024),
                    disk_io_read_mb=disk_read,
                    disk_io_write_mb=disk_write,
                    network_bytes_sent=net_sent,
                    network_bytes_recv=net_recv,
                    timestamp=time.time()
                )
                
                with self._lock:
                    self._metrics.append(metrics)
                    self._detect_bottlenecks(metrics)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(interval)
    
    def _detect_bottlenecks(self, metrics: PerformanceMetrics):
        """Detect performance bottlenecks automatically"""
        bottlenecks = []
        
        if metrics.cpu_percent > 90:
            bottlenecks.append(("CPU", f"High CPU usage: {metrics.cpu_percent:.1f}%"))
        
        if metrics.memory_percent > 85:
            bottlenecks.append(("Memory", f"High memory usage: {metrics.memory_percent:.1f}%"))
        
        if metrics.disk_io_read_mb > 100 or metrics.disk_io_write_mb > 100:
            bottlenecks.append(("IO", f"High disk IO: R{metrics.disk_io_read_mb:.1f}MB W{metrics.disk_io_write_mb:.1f}MB"))
        
        if bottlenecks:
            with self._lock:
                self._bottlenecks.extend(bottlenecks)
                if len(self._bottlenecks) > 100:  # Keep only recent bottlenecks
                    self._bottlenecks = self._bottlenecks[-50:]
    
    def profile_function(self, func_name: str = None):
        """Decorator for profiling function execution"""
        def decorator(func):
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    elapsed = time.time() - start_time
                    with self._lock:
                        stats = self._function_stats[name]
                        stats["calls"] += 1
                        stats["total_time"] += elapsed
                        stats["avg_time"] = stats["total_time"] / stats["calls"]
            return wrapper
        return decorator
    
    def track_endpoint(self, endpoint: str, duration: float, error: bool = False):
        """Track API endpoint performance"""
        with self._lock:
            stats = self._endpoint_stats[endpoint]
            stats["requests"] += 1
            stats["total_time"] += duration
            if error:
                stats["errors"] += 1
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get latest performance metrics"""
        with self._lock:
            return self._metrics[-1] if self._metrics else None
    
    def get_metrics_summary(self, duration_seconds: int = 300) -> Dict[str, Any]:
        """Get performance summary for last N seconds"""
        cutoff_time = time.time() - duration_seconds
        
        with self._lock:
            recent_metrics = [m for m in self._metrics if m.timestamp >= cutoff_time]
            
            if not recent_metrics:
                return {"error": "No metrics available"}
            
            # Calculate averages
            avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
            max_cpu = max(m.cpu_percent for m in recent_metrics)
            max_memory = max(m.memory_percent for m in recent_metrics)
            
            # Function performance
            top_functions = sorted(
                [(name, stats) for name, stats in self._function_stats.items()],
                key=lambda x: x[1]["avg_time"],
                reverse=True
            )[:10]
            
            # Endpoint performance
            endpoint_summary = {}
            for endpoint, stats in self._endpoint_stats.items():
                if stats["requests"] > 0:
                    endpoint_summary[endpoint] = {
                        "requests": stats["requests"],
                        "avg_response_time": stats["total_time"] / stats["requests"],
                        "error_rate": stats["errors"] / stats["requests"] * 100
                    }
            
            return {
                "period_seconds": duration_seconds,
                "samples_count": len(recent_metrics),
                "system_performance": {
                    "cpu_avg": avg_cpu,
                    "cpu_max": max_cpu,
                    "memory_avg": avg_memory,
                    "memory_max": max_memory,
                    "current_memory_mb": recent_metrics[-1].memory_mb
                },
                "top_slow_functions": top_functions[:5],
                "endpoint_performance": endpoint_summary,
                "recent_bottlenecks": self._bottlenecks[-10:],
                "bottleneck_count": len(self._bottlenecks)
            }
    
    def export_metrics(self, filepath: str, duration_seconds: int = 3600):
        """Export performance data to JSON"""
        cutoff_time = time.time() - duration_seconds
        
        with self._lock:
            recent_metrics = [asdict(m) for m in self._metrics if m.timestamp >= cutoff_time]
            
            export_data = {
                "exported_at": time.time(),
                "duration_seconds": duration_seconds,
                "metrics": recent_metrics,
                "function_stats": dict(self._function_stats),
                "endpoint_stats": dict(self._endpoint_stats),
                "bottlenecks": self._bottlenecks
            }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Performance metrics exported to {filepath}")

class BottleneckDetector:
    """Real-time bottleneck detection and alerting"""
    
    def __init__(self, profiler: PerformanceProfiler):
        self.profiler = profiler
        self._alerts = []
        self._lock = threading.RLock()
    
    def check_performance_issues(self) -> List[Dict[str, Any]]:
        """Check for current performance issues"""
        current = self.profiler.get_current_metrics()
        if not current:
            return []
        
        issues = []
        
        # CPU bottleneck
        if current.cpu_percent > 80:
            severity = "critical" if current.cpu_percent > 95 else "warning"
            issues.append({
                "type": "cpu_bottleneck",
                "severity": severity,
                "value": current.cpu_percent,
                "message": f"High CPU usage: {current.cpu_percent:.1f}%",
                "timestamp": current.timestamp
            })
        
        # Memory bottleneck
        if current.memory_percent > 80:
            severity = "critical" if current.memory_percent > 95 else "warning"
            issues.append({
                "type": "memory_bottleneck", 
                "severity": severity,
                "value": current.memory_percent,
                "message": f"High memory usage: {current.memory_percent:.1f}%",
                "timestamp": current.timestamp
            })
        
        # Function performance issues
        summary = self.profiler.get_metrics_summary(60)  # Last minute
        if "top_slow_functions" in summary:
            for func_name, stats in summary["top_slow_functions"][:3]:
                if stats["avg_time"] > 1.0:  # Functions taking >1s
                    issues.append({
                        "type": "slow_function",
                        "severity": "warning",
                        "function": func_name,
                        "avg_time": stats["avg_time"],
                        "message": f"Slow function {func_name}: {stats['avg_time']:.2f}s avg",
                        "timestamp": current.timestamp
                    })
        
        return issues

# Global profiler instance
_global_profiler: Optional[PerformanceProfiler] = None

def get_profiler() -> PerformanceProfiler:
    """Get global profiler instance"""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
        _global_profiler.start_monitoring()
    return _global_profiler

def profile(func_name: str = None):
    """Decorator for profiling functions"""
    return get_profiler().profile_function(func_name)

def track_endpoint_performance(endpoint: str, start_time: float, error: bool = False):
    """Track endpoint performance metrics"""
    duration = time.time() - start_time
    get_profiler().track_endpoint(endpoint, duration, error) 