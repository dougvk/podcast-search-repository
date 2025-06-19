#!/usr/bin/env python3

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from memvid import MemvidRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("transcript-search-20")

# Global variables
retriever = None
episode_mapping = {}

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(20, ge=1, le=100)
    threshold: float = Field(0.0, ge=0.0, le=1.0)

class EpisodeMetadata(BaseModel):
    title: Optional[str] = None
    filename: str
    speaker: Optional[str] = None
    timestamp: Optional[str] = None

class SearchResult(BaseModel):
    text: str
    score: float
    episode: EpisodeMetadata

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int

# Episode metadata functions
def load_episode_mapping():
    """Load episode mapping for 20 random episodes"""
    global episode_mapping
    project_root = Path(__file__).parent
    mapping_file = project_root / "data" / "episode_mapping_20.json"
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
        logger.warning(f"❌ No episode mapping found at {mapping_file.absolute()}")

def extract_frame_from_text(text):
    """Extract frame number by looking up text in memvid index"""
    try:
        project_root = Path(__file__).parent
        index_file = project_root / "data" / "podcast_batch_20_random_index.json"
        
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
    if not episode_mapping:
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
    
    if episode_data:
        logger.debug(f"Getting episode metadata for frame {frame_number}")
        logger.debug(f"Found episode: {episode_data.get('title', 'Unknown')}")
        return EpisodeMetadata(
            filename=episode_data.get("filename", "transcript.txt"),
            title=episode_data.get("title"),
            speaker=episode_data.get("speaker"),
            timestamp=episode_data.get("timestamp")
        )
    
    # Fallback
    return EpisodeMetadata(
        filename="transcript.txt",
        title=None,
        speaker=None,
        timestamp=None
    )

def initialize_search_engine():
    """Initialize the search engine with 20 random episodes data"""
    global retriever
    
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    
    # Look for the 20 random episodes dataset
    video_file = data_dir / "podcast_batch_20_random.mp4"
    index_file = data_dir / "podcast_batch_20_random_index.json"
    
    logger.info(f"Initializing search engine from: {data_dir}")
    logger.info(f"Looking for video: {video_file}")
    logger.info(f"Looking for index: {index_file}")
    
    if video_file.exists() and index_file.exists():
        retriever = MemvidRetriever(str(video_file), str(index_file))
        load_episode_mapping()
        logger.info("✅ Search engine initialized successfully")
        return True
    else:
        logger.error(f"❌ Required files not found:")
        logger.error(f"   Video exists: {video_file.exists()}")
        logger.error(f"   Index exists: {index_file.exists()}")
        return False

# FastAPI app
app = FastAPI(
    title="Transcript Search API - 20 Episodes",
    description="Search through 20 random podcast episodes with episode metadata",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Initialize the search engine on startup"""
    if not initialize_search_engine():
        logger.error("Failed to initialize search engine")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    is_healthy = retriever is not None
    logger.info(f"Health check: {'healthy' if is_healthy else 'unhealthy'} (search_engine={is_healthy})")
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "search_engine_initialized": is_healthy,
        "episode_mapping_entries": len(episode_mapping)
    }

@app.post("/search", response_model=SearchResponse)
async def search_transcripts(request: SearchRequest):
    """Search through transcript content with episode metadata"""
    if not retriever:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    logger.info(f"Search query: '{request.query}' (limit={request.limit})")
    
    try:
        # Perform search
        results = retriever.search(request.query, top_k=request.limit)
        
        # Convert results and add episode metadata
        search_results = []
        for i, result_text in enumerate(results):
            # Extract frame number from text
            frame_number = extract_frame_from_text(result_text)
            
            # Get episode metadata
            episode_metadata = get_episode_metadata(frame_number)
            
            search_results.append(SearchResult(
                text=result_text,
                score=1.0 - (i * 0.1),  # Simple scoring based on rank
                episode=episode_metadata
            ))
        
        return SearchResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results)
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/info")
async def system_info():
    """System information endpoint"""
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    
    return {
        "service": "transcript-search-20-episodes",
        "version": "1.0.0",
        "search_engine_initialized": retriever is not None,
        "episode_mapping_entries": len(episode_mapping),
        "data_directory": str(data_dir.absolute()),
        "dataset": "podcast_batch_20_random"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info") 