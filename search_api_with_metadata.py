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
logger = logging.getLogger("transcript-search-metadata")

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
    total_found: int

def load_episode_mapping():
    """Load episode mapping"""
    global episode_mapping
    project_root = Path(__file__).parent
    mapping_file = project_root / "data" / "episode_mapping.json"
    
    logger.info(f"Loading episode mapping from: {mapping_file.absolute()}")
    if mapping_file.exists():
        try:
            with open(mapping_file, 'r') as f:
                episode_mapping = json.load(f)
            logger.info(f"✅ Loaded episode mapping with {len(episode_mapping)} entries")
        except Exception as e:
            logger.warning(f"Failed to load episode mapping: {e}")
            episode_mapping = {}
    else:
        logger.warning(f"❌ No episode mapping found - using generic metadata")

def extract_frame_from_text(text):
    """Extract frame number by looking up text in memvid index"""
    try:
        project_root = Path(__file__).parent
        index_file = project_root / "data" / "podcast_batch_001_index.json"
        
        if index_file.exists():
            with open(index_file, 'r') as f:
                index_data = json.load(f)
            
            if 'metadata' in index_data:
                for item in index_data['metadata']:
                    if item.get('text', '').strip().startswith(text.strip()[:100]):
                        frame = item.get('frame', 0)
                        logger.info(f"Frame extracted: {frame} for text: {text[:50]}...")
                        return frame
        return 0
    except Exception as e:
        logger.warning(f"Error extracting frame: {e}")
        return 0

def get_episode_metadata(frame_number):
    """Get episode metadata for a frame number"""
    logger.info(f"Getting episode metadata for frame {frame_number}")
    
    if not episode_mapping:
        logger.info("No episode mapping - returning generic metadata")
        return EpisodeMetadata(
            filename="transcript.txt",
            title=None,
            speaker=None,
            timestamp=None
        )
    
    # Try exact frame match first
    episode_data = episode_mapping.get(str(frame_number))
    if not episode_data:
        # Find closest frame
        frame_keys = [int(k) for k in episode_mapping.keys() if k.isdigit()]
        if frame_keys:
            closest_frame = min(frame_keys, key=lambda x: abs(x - frame_number))
            episode_data = episode_mapping.get(str(closest_frame))
            logger.info(f"Used closest frame {closest_frame} for requested frame {frame_number}")
    
    if episode_data:
        logger.info(f"Found episode: {episode_data.get('title', 'Unknown')}")
        return EpisodeMetadata(
            filename=episode_data.get("filename", "transcript.txt"),
            title=episode_data.get("title"),
            speaker=episode_data.get("speaker"),
            timestamp=episode_data.get("timestamp")
        )
    
    logger.info("No episode data found - returning generic metadata")
    return EpisodeMetadata(
        filename="transcript.txt",
        title=None,
        speaker=None,
        timestamp=None
    )

def initialize_search_engine():
    """Initialize the search engine"""
    global retriever
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    
    logger.info(f"Initializing search engine from: {data_dir.absolute()}")
    
    video_files = list(data_dir.glob("*.mp4")) + list(data_dir.glob("*.avi"))
    if video_files:
        video_file = video_files[0]
        base_name = video_file.stem
        index_file = data_dir / f"{base_name}_index.json"
        
        logger.info(f"Found video: {video_file}")
        logger.info(f"Looking for index: {index_file}")
        
        if index_file.exists():
            retriever = MemvidRetriever(str(video_file), str(index_file))
            load_episode_mapping()
            logger.info("✅ Search engine initialized successfully")
            return True
    return False

# Create FastAPI app
app = FastAPI(title="Transcript Search with Episode Metadata")

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    if not initialize_search_engine():
        logger.error("Failed to initialize search engine")

@app.post("/search", response_model=SearchResponse)
async def search_with_metadata(request: SearchRequest):
    """Search with episode metadata"""
    logger.info(f"Search query: '{request.query}' (limit={request.limit})")
    
    if not retriever:
        if not initialize_search_engine():
            raise HTTPException(status_code=503, detail="Search engine not available")
    
    # Perform search
    raw_results = retriever.search(request.query, top_k=request.limit)
    
    # Convert to our format with episode metadata
    results = []
    for result in raw_results:
        if isinstance(result, str):
            text = result
            score = 1.0
        else:
            text, score = result
        
        if score >= request.threshold:
            # Extract frame and get episode metadata
            frame_number = extract_frame_from_text(text)
            episode_meta = get_episode_metadata(frame_number)
            
            results.append(SearchResult(
                text=text,
                score=float(score),
                episode=episode_meta
            ))
    
    return SearchResponse(
        query=request.query,
        results=results[:request.limit],
        total_found=len(results)
    )

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy", 
        "search_engine_initialized": retriever is not None,
        "episode_mapping_entries": len(episode_mapping)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 