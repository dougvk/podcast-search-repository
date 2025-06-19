#!/usr/bin/env python3
import json
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from memvid import MemvidRetriever
import re

app = FastAPI()
retriever = None
episode_mapping = {}

class SearchRequest(BaseModel):
    query: str
    limit: int = 20

def discover_data():
    """Auto-discover the latest index and mapping files"""
    data_dir = Path("data")
    
    # Prefer selected_episodes if it exists
    if (data_dir / "selected_episodes.mp4").exists():
        video_file = data_dir / "selected_episodes.mp4"
        index_file = data_dir / "selected_episodes_index"
    else:
        # Find any video files
        video_files = list(data_dir.glob("*.mp4"))
        if not video_files:
            raise Exception("No video files found")
        
        video_file = video_files[0]  # Use first found
        index_file = data_dir / (video_file.stem + "_index")
    
    # Find episode mapping
    mapping_files = list(data_dir.glob("episode_mapping*.json"))
    mapping_file = mapping_files[0] if mapping_files else None
    
    return video_file, index_file, mapping_file

def extract_frame_from_text(text):
    """Extract frame number from text"""
    match = re.search(r'Frame (\d+)', text)
    return int(match.group(1)) if match else None

@app.on_event("startup")
async def startup():
    global retriever, episode_mapping
    
    video_file, index_file, mapping_file = discover_data()
    
    print(f"Loading: {video_file}")
    print(f"Index: {index_file}")
    print(f"Mapping: {mapping_file}")
    
    retriever = MemvidRetriever(str(video_file), str(index_file))
    
    if mapping_file:
        with open(mapping_file) as f:
            episode_mapping = json.load(f)
        print(f"✅ Loaded {len(episode_mapping)} episode mappings")

@app.get("/health")
def health():
    return {"status": "healthy", "engine": retriever is not None}

@app.post("/search")
def search(request: SearchRequest):
    if not retriever:
        raise HTTPException(500, "Search engine not initialized")
    
    results = retriever.search(request.query, top_k=request.limit)
    
    search_results = []
    for i, text in enumerate(results):
        frame_num = extract_frame_from_text(text)
        episode_info = episode_mapping.get(str(frame_num), {
            "title": "Unknown Episode",
            "filename": "unknown.txt",
            "speaker": "Unknown",
            "timestamp": None
        })
        
        search_results.append({
            "text": text,
            "score": 1.0 - (i * 0.1),  # Fake score
            "episode": episode_info
        })
    
    return {"results": search_results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 