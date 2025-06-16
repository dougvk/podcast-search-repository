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
from memvid import MemvidRetriever
from core.models import Episode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("transcript-search")

# Create FastAPI app
app = FastAPI(
    title="Transcript Search API",
    description="Natural language search for transcript chunks",
    version="1.0.0"
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    correlation_id = str(uuid.uuid4())
    request.state.correlation_id = correlation_id
    
    start_time = time.time()
    logger.info(f"[{correlation_id}] {request.method} {request.url}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"[{correlation_id}] {response.status_code} - {process_time:.3f}s")
    
    return response

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

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service health status")
    service: str = Field(..., description="Service name")
    timestamp: str = Field(..., description="Health check timestamp")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")

# Global search engine instance
retriever = None

def initialize_search_engine():
    """Initialize the search engine with available data"""
    global retriever
    data_dir = Path("data")
    
    # Look for memvid files
    video_files = list(data_dir.glob("*.mp4")) + list(data_dir.glob("*.avi"))
    if video_files:
        video_file = video_files[0]
        index_file = video_file.with_suffix("_index.json")
        if index_file.exists():
            retriever = MemvidRetriever(str(video_file), str(index_file))
            return True
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
        
        # Perform search
        raw_results = retriever.search(request.query, top_k=request.limit * 2)  # Get extra for deduplication
        
        # Convert to our format and filter by threshold
        formatted_results = []
        for text, score in raw_results:
            if score >= request.threshold:
                # Extract episode metadata (simplified)
                episode_meta = EpisodeMetadata(
                    filename="transcript.txt",  # Default for now
                    title=None,
                    speaker=None,
                    timestamp=None
                )
                
                formatted_results.append({
                    'text': text,
                    'score': float(score),
                    'episode': episode_meta
                })
        
        # Deduplicate results
        unique_results = deduplicate_results(formatted_results)
        
        # Limit to requested amount
        final_results = unique_results[:request.limit]
        
        execution_time = (time.time() - start_time) * 1000  # Convert to ms
        
        logger.info(f"[{correlation_id}] Search completed: {len(final_results)} results in {execution_time:.1f}ms")
        
        return SearchResponse(
            query=request.query,
            results=[SearchResult(**result) for result in final_results],
            total_found=len(formatted_results),
            execution_time_ms=execution_time
        )
        
    except HTTPException:
        raise  # Re-raise HTTPExceptions as-is
    except Exception as e:
        logger.error(f"[{correlation_id}] Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

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

@api_router.get("/info")
async def system_info():
    """System information endpoint"""
    data_dir = Path("data")
    video_files = list(data_dir.glob("*.mp4")) + list(data_dir.glob("*.avi"))
    
    return {
        "service": "transcript-search",
        "version": "1.0.0",
        "search_engine_initialized": retriever is not None,
        "available_videos": len(video_files),
        "data_directory": str(data_dir.absolute())
    }

# Include router
app.include_router(api_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 