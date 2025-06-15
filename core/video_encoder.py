"""Video encoding for text storage using memvid."""
from typing import List, Dict, Any
from pathlib import Path
from memvid import MemvidEncoder
import os

class VideoEncoder:
    def __init__(self, output_dir: str = "data/video_libraries"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def encode(self, chunks: List[str], episode_id: str) -> Dict[str, str]:
        """Encode text chunks to video."""
        encoder = MemvidEncoder()
        encoder.add_chunks(chunks)
        
        video_path = str(self.output_dir / f"{episode_id}.mp4")
        index_path = str(self.output_dir / f"{episode_id}_index.json")
        
        encoder.build_video(video_path, index_path)
        
        return {
            "video_path": video_path,
            "index_path": index_path,
            "chunk_count": len(chunks)
        }
    
    def decode(self, video_path: str, index_path: str) -> List[str]:
        """Decode video back to text chunks."""
        from memvid import MemvidRetriever
        
        retriever = MemvidRetriever(video_path, index_path)
        # Get all chunks by searching broadly
        results = retriever.search("", top_k=1000)  # Get all chunks with empty query
        # Results are likely just strings, not dicts
        return results if isinstance(results, list) else [results]