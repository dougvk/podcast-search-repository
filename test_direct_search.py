#!/usr/bin/env python3

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from memvid import MemvidRetriever

def load_episode_mapping():
    """Load episode mapping"""
    mapping_file = project_root / "data" / "episode_mapping.json"
    if mapping_file.exists():
        with open(mapping_file, 'r') as f:
            return json.load(f)
    return {}

def extract_frame_from_text(text, index_data):
    """Extract frame number by looking up text in memvid index"""
    try:
        if 'metadata' in index_data:
            for item in index_data['metadata']:
                if item.get('text', '').strip().startswith(text.strip()[:100]):
                    return item.get('frame', 0)
        return 0
    except Exception:
        return 0

def get_episode_metadata(frame_number, episode_mapping):
    """Get episode metadata for a frame number"""
    if not episode_mapping:
        return {"title": None, "filename": "transcript.txt", "speaker": None, "timestamp": None}
    
    # Try exact frame match first
    episode_data = episode_mapping.get(str(frame_number))
    if not episode_data:
        # Find closest frame
        frame_keys = [int(k) for k in episode_mapping.keys() if k.isdigit()]
        if frame_keys:
            closest_frame = min(frame_keys, key=lambda x: abs(x - frame_number))
            episode_data = episode_mapping.get(str(closest_frame))
    
    if episode_data:
        return {
            "title": episode_data.get("title"),
            "filename": episode_data.get("filename", "transcript.txt"),
            "speaker": episode_data.get("speaker"),
            "timestamp": episode_data.get("timestamp")
        }
    
    return {"title": None, "filename": "transcript.txt", "speaker": None, "timestamp": None}

def main():
    print("Testing direct search with episode mapping...")
    
    # Load episode mapping
    episode_mapping = load_episode_mapping()
    print(f"Loaded episode mapping with {len(episode_mapping)} entries")
    
    # Load index data for frame extraction
    index_file = project_root / "data" / "podcast_batch_001_index.json"
    if not index_file.exists():
        print(f"Index file not found: {index_file}")
        return
    
    with open(index_file, 'r') as f:
        index_data = json.load(f)
    print(f"Loaded index with {len(index_data.get('metadata', []))} metadata entries")
    
    # Initialize memvid retriever
    video_file = project_root / "data" / "podcast_batch_001.mp4"
    if not video_file.exists():
        print(f"Video file not found: {video_file}")
        return
    
    retriever = MemvidRetriever(str(video_file), str(index_file))
    print("Memvid retriever initialized")
    
    # Perform search
    query = "Boris Johnson"
    print(f"\nSearching for: '{query}'")
    
    raw_results = retriever.search(query, top_k=2)
    print(f"Found {len(raw_results)} raw results")
    
    # Process results with episode metadata
    for i, result in enumerate(raw_results):
        if isinstance(result, str):
            text = result
            score = 1.0
        else:
            text, score = result
        
        frame_number = extract_frame_from_text(text, index_data)
        episode_meta = get_episode_metadata(frame_number, episode_mapping)
        
        print(f"\nResult {i+1}:")
        print(f"  Text: {text[:100]}...")
        print(f"  Frame: {frame_number}")
        print(f"  Episode: {episode_meta['title']}")
        print(f"  Filename: {episode_meta['filename']}")

if __name__ == "__main__":
    main() 