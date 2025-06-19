from fastapi import FastAPI, APIRouter, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
import uvicorn
import time
import os
import logging
import uuid
from pathlib import Path
from datetime import datetime
from memvid import MemvidRetriever
from core.models import Episode
from .cache_manager import get_cache_manager, CacheConfig
from .index_optimizer import IndexOptimizer, IndexConfig, create_optimized_index
from .embedding_manager import get_embedding_manager
from .async_handler import get_async_handler, ConcurrentSearchProcessor
from .performance_monitor import get_profiler, BottleneckDetector, track_endpoint_performance, profile
from .system_tuner import get_system_tuner, auto_tune_system, optimize_for_search
from .monitoring import monitoring_middleware, monitor_search, get_metrics, health_check as monitoring_health
import json

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("transcript-search")

# Create FastAPI app
app = FastAPI(
    title="Transcript Search API",
    description="Natural language search for transcript chunks",
    version="1.0.0"
)

# Initialize cache manager, index optimizer, embedding manager, async handler, and profiler
cache = get_cache_manager()
index_optimizer = None
embedding_manager = get_embedding_manager()
async_handler = get_async_handler()
profiler = get_profiler()
bottleneck_detector = BottleneckDetector(profiler)
system_tuner = get_system_tuner(profiler)

# Add monitoring middleware
app.middleware("http")(monitoring_middleware)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Pydantic models
class SearchRequest(BaseModel):
    """Search request model"""
    query: str = Field(..., min_length=1, max_length=1000, description="Natural language search query")
    limit: int = Field(20, ge=1, le=100, description="Maximum number of results to return")
    threshold: float = Field(0.0, ge=0.0, le=1.0, description="Minimum relevance score threshold")
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        return v.strip()

class BatchSearchRequest(BaseModel):
    """Batch search request model"""
    queries: List[str] = Field(..., min_length=1, max_length=50, description="List of search queries")
    limit: int = Field(20, ge=1, le=100, description="Maximum results per query")
    threshold: float = Field(0.0, ge=0.0, le=1.0, description="Minimum relevance score threshold")
    
    @field_validator('queries')
    @classmethod
    def validate_queries(cls, v):
        return [q.strip() for q in v if q.strip()]

class EpisodeMetadata(BaseModel):
    """Episode metadata model"""
    title: Optional[str] = None
    filename: str
    speaker: Optional[str] = None
    timestamp: Optional[str] = None

class SearchResult(BaseModel):
    """Individual search result model"""
    text: str = Field(..., description="Relevant text chunk")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    episode: EpisodeMetadata = Field(..., description="Episode metadata")

class SearchResponse(BaseModel):
    """Search response model"""
    query: str = Field(..., description="Original search query")
    results: List[SearchResult] = Field(..., description="Search results")
    total_found: int = Field(..., ge=0, description="Total number of results found")
    execution_time_ms: float = Field(..., ge=0, description="Query execution time in milliseconds")
    cache_hit: bool = Field(default=False, description="Whether result came from cache")
    correlation_id: str = Field(description="Request correlation ID")

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service health status")
    service: str = Field(..., description="Service name")
    timestamp: str = Field(..., description="Health check timestamp")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")

# Global variables
retriever = None
episode_mapping = {}

def load_episode_mapping():
    """Load episode mapping if available"""
    global episode_mapping
    
    # Use the same project root logic as initialize_search_engine
    project_root = Path(__file__).parent.parent
    mapping_file = project_root / "data" / "episode_mapping.json"
    
    logger.info(f"Looking for episode mapping at: {mapping_file.absolute()}")
    if mapping_file.exists():
        try:
            with open(mapping_file, 'r') as f:
                episode_mapping = json.load(f)
            logger.info(f"✅ Loaded episode mapping with {len(episode_mapping)} entries")
        except Exception as e:
            logger.warning(f"Failed to load episode mapping: {e}")
            episode_mapping = {}
    else:
        logger.warning(f"❌ No episode mapping found at {mapping_file.absolute()} - using generic metadata")

def extract_frame_from_text(text):
    """Extract frame number by looking up text in memvid index"""
    try:
        # Use the same project root logic
        project_root = Path(__file__).parent.parent
        index_file = project_root / "data" / "podcast_batch_001_index.json"
        
        logger.debug(f"Looking for index at: {index_file.absolute()}")
        if index_file.exists():
            with open(index_file, 'r') as f:
                index_data = json.load(f)
            
            # Search for matching text in index metadata
            if 'metadata' in index_data:
                for item in index_data['metadata']:
                    if item.get('text', '').strip().startswith(text.strip()[:100]):
                        frame = item.get('frame', 0)
                        logger.debug(f"Frame extracted: {frame} for text: {text[:50]}...")
                        return frame
        logger.debug(f"No frame found for text: {text[:50]}...")
        return 0
    except Exception as e:
        logger.warning(f"Error extracting frame: {e}")
        return 0

def get_episode_metadata(frame_number):
    """Get episode metadata for a frame number"""
    logger.debug(f"Getting episode metadata for frame {frame_number}, mapping has {len(episode_mapping)} entries")
    
    if not episode_mapping:
        logger.debug("No episode mapping available - returning generic metadata")
        return EpisodeMetadata(
            filename="transcript.txt",
            title=None,
            speaker=None,
            timestamp=None
        )
    
    # Try exact frame match first, then approximate
    episode_data = episode_mapping.get(str(frame_number))
    if not episode_data:
        # Find closest frame (simple fallback)
        frame_keys = [int(k) for k in episode_mapping.keys() if k.isdigit()]
        if frame_keys:
            closest_frame = min(frame_keys, key=lambda x: abs(x - frame_number))
            episode_data = episode_mapping.get(str(closest_frame))
            logger.debug(f"Used closest frame {closest_frame} for requested frame {frame_number}")
    
    if episode_data:
        logger.debug(f"Found episode data: {episode_data.get('title', 'Unknown')}")
        return EpisodeMetadata(
            filename=episode_data.get("filename", "transcript.txt"),
            title=episode_data.get("title"),
            speaker=episode_data.get("speaker"),
            timestamp=episode_data.get("timestamp")
        )
    
    # Fallback to generic
    logger.debug(f"No episode data found for frame {frame_number} - returning generic metadata")
    return EpisodeMetadata(
        filename="transcript.txt",
        title=None,
        speaker=None,
        timestamp=None
    )

def initialize_search_engine():
    """Initialize the search engine with available data"""
    global retriever
    
    # Determine the correct project root directory
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    
    logger.info(f"Initializing search engine with data directory: {data_dir.absolute()}")
    
    # Look for memvid files
    video_files = list(data_dir.glob("*.mp4")) + list(data_dir.glob("*.avi"))
    if video_files:
        video_file = video_files[0]
        # Fix: Look for _index.json files that match the video basename
        base_name = video_file.stem
        index_file = data_dir / f"{base_name}_index.json"
        
        logger.info(f"Found video: {video_file}")
        logger.info(f"Looking for index: {index_file}")
        
        if index_file.exists():
            retriever = MemvidRetriever(str(video_file), str(index_file))
            load_episode_mapping()  # **NEW: Load episode mapping**
            logger.info("✅ Search engine initialized successfully")
            return True
        else:
            logger.warning(f"Index file not found: {index_file}")
    else:
        logger.warning(f"No video files found in: {data_dir}")
    return False

def deduplicate_results(results: List[Dict], threshold: float = 0.9) -> List[Dict]:
    """Remove similar results based on text similarity"""
    if not results:
        return results
    
    unique_results = []
    for result in results:
        is_duplicate = False
        for unique in unique_results:
            # Simple similarity check based on text overlap
            text1_words = set(result['text'].lower().split())
            text2_words = set(unique['text'].lower().split())
            if text1_words and text2_words:
                overlap = len(text1_words & text2_words) / len(text1_words | text2_words)
                if overlap > threshold:
                    is_duplicate = True
                    break
        if not is_duplicate:
            unique_results.append(result)
    
    return unique_results

# Create API router
api_router = APIRouter(prefix="/api/v1")

@api_router.post("/search", response_model=SearchResponse)
@monitor_search("semantic")
async def search_transcripts(request: SearchRequest, http_request: Request):
    """Search transcript chunks using natural language queries"""
    correlation_id = getattr(http_request.state, 'correlation_id', 'unknown')
    start_time = time.time()
    
    logger.info(f"[{correlation_id}] Search query: '{request.query}' (limit={request.limit}, threshold={request.threshold})")
    
    try:
        if not retriever:
            if not initialize_search_engine():
                logger.error(f"[{correlation_id}] Search engine initialization failed")
                raise HTTPException(status_code=503, detail="Search engine not available")
        
        # Check cache first
        cached_results = cache.get_search_results(request.query, request.limit, request.threshold)
        cache_hit = cached_results is not None
        
        if cache_hit:
            final_results = cached_results
            formatted_results = cached_results  # For total_found calculation
            logger.info(f"[{correlation_id}] Cache HIT")
        else:
            # Perform search
            raw_results = retriever.search(request.query, top_k=request.limit * 2)  # Get extra for deduplication
            
            # Convert to our format and filter by threshold
            formatted_results = []
            for result in raw_results:
                # Handle memvid results which are just text strings
                if isinstance(result, str):
                    text = result
                    score = 1.0  # Default score since memvid doesn't return scores
                    # **NEW: Extract frame number from memvid index by matching text**
                    frame_number = extract_frame_from_text(text)
                else:
                    # Handle tuple format (text, score) if memvid changes
                    text, score = result
                    frame_number = extract_frame_from_text(text)
                
                if score >= request.threshold:
                    # **NEW: Get actual episode metadata using frame number**
                    episode_meta = get_episode_metadata(frame_number)
                    
                    formatted_results.append({
                        'text': text,
                        'score': float(score),
                        'episode': episode_meta
                    })
            
            # Deduplicate results
            unique_results = deduplicate_results(formatted_results)
            
            # Limit to requested amount
            final_results = unique_results[:request.limit]
            
            # Cache the results (final_results are already dicts)
            cache.cache_search_results(request.query, final_results, request.limit, request.threshold)
            logger.info(f"[{correlation_id}] Cache MISS - results cached")
        
        execution_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Track endpoint performance
        track_endpoint_performance("/search", start_time, error=False)
        
        logger.info(f"[{correlation_id}] Search completed: {len(final_results)} results in {execution_time:.1f}ms")
        
        return SearchResponse(
            query=request.query,
            results=[SearchResult(**result) for result in final_results],
            total_found=len(formatted_results),
            execution_time_ms=execution_time,
            cache_hit=cache_hit,
            correlation_id=correlation_id
        )
        
    except HTTPException:
        track_endpoint_performance("/search", start_time, error=True)
        raise  # Re-raise HTTPExceptions as-is
    except Exception as e:
        track_endpoint_performance("/search", start_time, error=True)
        logger.error(f"[{correlation_id}] Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@api_router.post("/search/simple", response_model=SearchResponse)
@monitor_search("simple")
async def simple_search(request: SearchRequest, http_request: Request):
    """
    Simple direct semantic search optimized for long natural language questions.
    Bypasses all preprocessing complexity - perfect for corpus queries.
    """
    correlation_id = getattr(http_request.state, 'correlation_id', 'unknown')
    start_time = time.time()
    
    logger.info(f"[{correlation_id}] Simple search query: '{request.query}' (limit={request.limit})")
    
    try:
        if not retriever:
            if not initialize_search_engine():
                logger.error(f"[{correlation_id}] Search engine initialization failed")
                raise HTTPException(status_code=503, detail="Search engine not available")
        
        # Use direct search (semantic only, no preprocessing)
        raw_results = retriever.search(request.query, top_k=request.limit)
        
        # Convert to our format
        formatted_results = []
        for result in raw_results:
            # Handle memvid results which are just text strings
            if isinstance(result, str):
                text = result
                score = 1.0  # Default score since memvid doesn't return scores
            else:
                # Handle tuple format if it exists
                try:
                    text, score = result
                except (ValueError, TypeError):
                    text = str(result)
                    score = 1.0
            
            if score >= request.threshold:
                # **NEW: Get actual episode metadata for simple search too**
                frame_number = extract_frame_from_text(text)
                episode_meta = get_episode_metadata(frame_number)
                
                formatted_results.append({
                    'text': text,
                    'score': float(score),
                    'episode': episode_meta
                })
        
        execution_time = (time.time() - start_time) * 1000
        
        logger.info(f"[{correlation_id}] Simple search completed: {len(formatted_results)} results in {execution_time:.1f}ms")
        
        return SearchResponse(
            query=request.query,
            results=[SearchResult(**result) for result in formatted_results],
            total_found=len(formatted_results),
            execution_time_ms=execution_time,
            cache_hit=False,  # Simple search doesn't use cache for maximum simplicity
            correlation_id=correlation_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{correlation_id}] Simple search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Simple search failed: {str(e)}")

@api_router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    from datetime import datetime
    import datetime as dt
    
    # Check search engine availability
    search_available = retriever is not None or initialize_search_engine()
    status = "healthy" if search_available else "degraded"
    
    logger.info(f"Health check: {status} (search_engine={search_available})")
    
    return HealthResponse(
        status=status,
        service="transcript-search",
        timestamp=datetime.now(dt.timezone.utc).isoformat()
    )

@api_router.get("/health/ready")
async def readiness_probe():
    """Readiness probe for Kubernetes"""
    search_available = retriever is not None or initialize_search_engine()
    if not search_available:
        raise HTTPException(status_code=503, detail="Search engine not ready")
    return {"status": "ready"}

@api_router.get("/health/live")
async def liveness_probe():
    """Liveness probe for Kubernetes"""
    return {"status": "alive"}

@api_router.get("/debug/paths")
async def debug_paths():
    """Debug endpoint to check file paths"""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    
    return {
        "timestamp": datetime.now().isoformat(),
        "__file__": str(Path(__file__).absolute()),
        "project_root": str(project_root.absolute()),
        "current_working_dir": str(Path.cwd()),
        "data_directory": str(data_dir.absolute()),
        "episode_mapping_exists": (project_root / "data" / "episode_mapping.json").exists(),
        "episode_mapping_size": len(episode_mapping),
        "index_exists": (project_root / "data" / "podcast_batch_001_index.json").exists(),
    }

@api_router.get("/info")
async def system_info():
    """System information endpoint"""
    # Use the same project root logic
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    video_files = list(data_dir.glob("*.mp4")) + list(data_dir.glob("*.avi"))
    
    return {
        "service": "transcript-search",
        "version": "1.0.0",
        "search_engine_initialized": retriever is not None,
        "available_videos": len(video_files),
        "data_directory": str(data_dir.absolute()),
        "debug_info": {
            "__file__": str(Path(__file__).absolute()),
            "project_root": str(project_root.absolute()),
            "current_working_dir": str(Path.cwd()),
            "episode_mapping_exists": (project_root / "data" / "episode_mapping.json").exists(),
            "episode_mapping_size": len(episode_mapping),
        }
    }

@api_router.get("/ready")
async def readiness_check():
    """Readiness check with cache status"""
    cache_stats = cache.stats()
    return {
        "status": "ready",
        "cache": {
            "memory_active": cache_stats["memory"]["active"],
            "redis_connected": cache_stats["redis"]["connected"]
        },
        "timestamp": datetime.now().isoformat()
    }

@api_router.get("/metrics")
async def system_metrics():
    """System metrics including cache performance"""
    return {
        "cache": cache.stats(),
        "timestamp": datetime.now().isoformat()
    }

@api_router.get("/metrics/prometheus")
async def prometheus_metrics():
    """Get Prometheus metrics"""
    return get_metrics()

@api_router.get("/health/monitoring")
async def monitoring_health():
    """Enhanced health check with monitoring metrics"""
    return monitoring_health()

@api_router.delete("/cache")
async def clear_cache():
    """Clear all cached data"""
    cache.clear()
    return {"status": "cache cleared", "timestamp": datetime.now().isoformat()}

@api_router.get("/index/optimize")
async def optimize_index():
    """Optimize FAISS index parameters for current dataset"""
    global index_optimizer
    
    if not retriever:
        raise HTTPException(status_code=503, detail="Search engine not available")
    
    try:
        # Get sample embeddings from retriever if available
        if hasattr(retriever, 'embeddings') and retriever.embeddings is not None:
            embeddings = retriever.embeddings
            
            if index_optimizer is None:
                index_optimizer = IndexOptimizer(dimension=embeddings.shape[1])
            
            # Auto-configure and optimize
            config = index_optimizer.auto_configure(len(embeddings), embeddings[:100])
            stats = index_optimizer.build_index(embeddings, config)
            
            # Optimize search parameters
            sample_queries = embeddings[:min(50, len(embeddings))]
            optimization_results = index_optimizer.optimize_search_params(sample_queries)
            
            return {
                "status": "optimized",
                "index_stats": stats,
                "optimization_results": optimization_results,
                "recommendations": {
                    "index_type": config.index_type.value,
                    "optimal_nprobe": optimization_results.get("optimized_nprobe"),
                    "estimated_speedup": f"{config.nlist / optimization_results.get('optimized_nprobe', 1):.1f}x"
                },
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=503, detail="No embeddings available for optimization")
            
    except Exception as e:
        logger.error(f"Index optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@api_router.get("/index/stats")
async def get_index_stats():
    """Get current index performance statistics"""
    global index_optimizer
    
    if index_optimizer is None:
        return {"status": "no_optimizer", "message": "Index optimizer not initialized"}
    
    stats = index_optimizer.get_stats()
    
    return {
        "index_stats": stats,
        "performance_metrics": {
            "searches_per_second": stats.get("searches", 0) / max(stats.get("optimization_time", 1), 1),
            "avg_build_time": stats.get("optimization_time", 0) / max(stats.get("builds", 1), 1),
            "total_operations": stats.get("searches", 0) + stats.get("builds", 0)
        },
        "timestamp": datetime.now().isoformat()
    }

@api_router.post("/index/tune")
async def tune_index_parameters(target_recall: float = 0.9):
    """Automatically tune index parameters for target recall"""
    global index_optimizer
    
    if index_optimizer is None or index_optimizer.index is None:
        raise HTTPException(status_code=503, detail="Index optimizer not available")
    
    if not (0.5 <= target_recall <= 1.0):
        raise HTTPException(status_code=400, detail="Target recall must be between 0.5 and 1.0")
    
    try:
        # Use sample queries from embeddings
        if hasattr(retriever, 'embeddings') and retriever.embeddings is not None:
            sample_queries = retriever.embeddings[:min(100, len(retriever.embeddings))]
            results = index_optimizer.optimize_search_params(sample_queries, target_recall)
            
            return {
                "status": "tuned",
                "target_recall": target_recall,
                "optimization_results": results,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=503, detail="No sample data available for tuning")
            
    except Exception as e:
        logger.error(f"Parameter tuning failed: {e}")
        raise HTTPException(status_code=500, detail=f"Tuning failed: {str(e)}")

@api_router.post("/search/batch")
@monitor_search("batch")
async def batch_search(request: BatchSearchRequest, http_request: Request):
    """Process multiple search queries concurrently"""
    correlation_id = getattr(http_request.state, 'correlation_id', 'unknown')
    start_time = time.time()
    
    logger.info(f"[{correlation_id}] Batch search: {len(request.queries)} queries")
    
    try:
        if not retriever:
            if not initialize_search_engine():
                raise HTTPException(status_code=503, detail="Search engine not available")
        
        # Define search function for concurrent processing
        def search_single(query: str) -> List[tuple]:
            return retriever.search(query, top_k=request.limit * 2)
        
        # Process queries concurrently
        processor = ConcurrentSearchProcessor(search_single, max_concurrent=10)
        concurrent_results = await processor.search_concurrent(request.queries)
        
        # Format all results
        formatted_responses = []
        for i, result in enumerate(concurrent_results):
            query = request.queries[i]
            
            if "error" in result:
                formatted_responses.append({
                    "query": query,
                    "results": [],
                    "total_found": 0,
                    "error": result["error"]
                })
            else:
                # Process search results
                raw_results = result["results"]
                formatted_results = []
                
                for text, score in raw_results:
                    if score >= request.threshold:
                        episode_meta = EpisodeMetadata(
                            filename="transcript.txt",
                            title=None,
                            speaker=None,
                            timestamp=None
                        )
                        formatted_results.append({
                            'text': text,
                            'score': float(score),
                            'episode': episode_meta
                        })
                
                # Deduplicate and limit results
                unique_results = deduplicate_results(formatted_results)
                final_results = unique_results[:request.limit]
                
                formatted_responses.append({
                    "query": query,
                    "results": [SearchResult(**result) for result in final_results],
                    "total_found": len(final_results)
                })
        
        execution_time = (time.time() - start_time) * 1000
        
        return {
            "batch_results": formatted_responses,
            "total_queries": len(request.queries),
            "execution_time_ms": execution_time,
            "correlation_id": correlation_id,
            "concurrent_stats": processor.get_stats()
        }
        
    except Exception as e:
        logger.error(f"[{correlation_id}] Batch search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch search failed: {str(e)}")

@api_router.get("/async/stats")
async def get_async_stats():
    """Get async handler performance statistics"""
    try:
        handler_stats = async_handler.get_stats()
        return {
            "async_handler": handler_stats,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Async stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get async stats: {str(e)}")

@api_router.get("/performance/metrics")
async def get_performance_metrics(duration: int = 300):
    """Get system performance metrics and bottleneck analysis"""
    try:
        summary = profiler.get_metrics_summary(duration)
        current_metrics = profiler.get_current_metrics()
        bottlenecks = bottleneck_detector.check_performance_issues()
        
        return {
            "performance_summary": summary,
            "current_metrics": current_metrics.__dict__ if current_metrics else None,
            "active_bottlenecks": bottlenecks,
            "monitoring_active": profiler._monitoring,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Performance metrics error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")

@api_router.post("/performance/export")
async def export_performance_data(duration: int = 3600, filename: Optional[str] = None):
    """Export performance data to file"""
    try:
        if not filename:
            filename = f"performance_export_{int(time.time())}.json"
        
        filepath = f"data/{filename}"
        os.makedirs("data", exist_ok=True)
        
        profiler.export_metrics(filepath, duration)
        
        return {
            "exported": True,
            "filepath": filepath,
            "duration_seconds": duration,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Performance export error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export performance data: {str(e)}")

@api_router.get("/system/tune")
async def auto_tune_system():
    """Auto-tune system parameters based on performance data"""
    try:
        metrics = profiler.get_metrics_summary(300)  # Last 5 minutes
        bottlenecks = bottleneck_detector.check_performance_issues()
        
        # Auto-tune based on current state
        recommendations = system_tuner.auto_tune(metrics, bottlenecks)
        
        return {
            "tuning_applied": recommendations,
            "current_config": system_tuner.get_current_config().__dict__,
            "performance_metrics": metrics,
            "bottlenecks_resolved": len([b for b in bottlenecks if b["severity"] == "resolved"]),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Auto-tune failed: {e}")
        raise HTTPException(status_code=500, detail=f"Auto-tune failed: {str(e)}")

@api_router.post("/system/optimize")
async def optimize_system(optimization_type: str = "search"):
    """Apply specific optimization profiles"""
    try:
        if optimization_type == "search":
            result = optimize_for_search(system_tuner, profiler)
        elif optimization_type == "batch":
            result = system_tuner.optimize_for_batch_processing()
        elif optimization_type == "memory":
            result = system_tuner.optimize_for_memory_efficiency()
        else:
            raise ValueError(f"Unknown optimization type: {optimization_type}")
        
        return {
            "optimization_applied": optimization_type,
            "changes": result,
            "new_config": system_tuner.get_current_config().__dict__,
            "estimated_improvement": system_tuner.estimate_performance_gain(),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"System optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@api_router.get("/system/status")
async def get_system_status():
    """Get comprehensive system health and configuration status"""
    try:
        current_metrics = profiler.get_current_metrics()
        bottlenecks = bottleneck_detector.check_performance_issues()
        config = system_tuner.get_current_config()
        
        # Calculate system health score
        health_score = system_tuner.calculate_health_score(current_metrics, bottlenecks)
        
        return {
            "health_score": health_score,
            "status": "optimal" if health_score > 0.8 else "degraded" if health_score > 0.5 else "critical",
            "current_metrics": current_metrics.__dict__ if current_metrics else None,
            "active_bottlenecks": bottlenecks,
            "system_config": config.__dict__,
            "uptime_seconds": profiler.get_uptime(),
            "last_tuning": system_tuner.get_last_tuning_time(),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"System status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@api_router.get("/embeddings/stats")
async def get_embedding_stats():
    """Get embedding compression and memory statistics"""
    try:
        stats = embedding_manager.get_stats()
        return {"embedding_stats": stats, "timestamp": time.time()}
    except Exception as e:
        logger.error(f"Embedding stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get embedding stats: {str(e)}")

@api_router.post("/embeddings/unload")
async def unload_embeddings(key: Optional[str] = None):
    """Unload embeddings from memory to free up space"""
    try:
        if key:
            embedding_manager.unload_embeddings(key)
            return {"message": f"Unloaded embeddings: {key}"}
        else:
            embedding_manager.unload_all()
            return {"message": "Unloaded all embeddings"}
    except Exception as e:
        logger.error(f"Embedding unload error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to unload embeddings: {str(e)}")

# Include router
app.include_router(api_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 