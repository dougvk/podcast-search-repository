"""Video encoding for text storage using memvid."""
from typing import List, Dict, Any, Optional
from pathlib import Path
from memvid import MemvidEncoder, MemvidRetriever
from memvid.config import get_default_config
from sentence_transformers import SentenceTransformer
import os
import logging

# Set tokenizers parallelism to false (memvid best practice)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Configure logging
logger = logging.getLogger(__name__)

class VideoEncoder:
    def __init__(self, 
                 output_dir: str = "data/video_libraries",
                 embedding_model: Optional[str] = None,
                 n_workers: int = 4,
                 fps: int = 30,
                 frame_size: int = 512,
                 video_codec: str = 'h264',
                 crf: int = 23):
        """Initialize VideoEncoder with memvid best practices.
        
        Args:
            output_dir: Directory for storing video files
            embedding_model: Custom embedding model name (e.g., 'all-MiniLM-L6-v2')
            n_workers: Number of workers for parallel processing
            fps: Frames per second for video encoding
            frame_size: Video frame size in pixels
            video_codec: Video codec ('h264', 'h265', 'mp4v')
            crf: Constant Rate Factor for compression quality (0-51, lower = better quality)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store embedding model preference
        self.embedding_model_name = embedding_model or 'sentence-transformers/all-MiniLM-L6-v2'
        
        # Store encoding parameters
        self.encoding_params = {
            'n_workers': n_workers,
            'fps': fps,
            'frame_size': frame_size,
            'video_codec': video_codec,
            'crf': crf
        }
        
        logger.info(f"VideoEncoder initialized with params: {self.encoding_params}")
        logger.info(f"Using embedding model: {self.embedding_model_name}")
        
    def _create_memvid_encoder(self) -> MemvidEncoder:
        """Create memvid encoder following proven best practices."""
        # Use memvid's proven configuration approach
        config = get_default_config()
        
        # Customize config for our use case
        if 'embedding' not in config:
            config['embedding'] = {}
        
        # Set custom embedding model
        config['embedding']['model'] = self.embedding_model_name
        
        # Set worker count
        config['n_workers'] = self.encoding_params['n_workers']
        
        # Create encoder with memvid's proven initialization
        return MemvidEncoder(config)
        
    def encode(self, chunks: List[str], episode_id: str) -> Dict[str, Any]:
        """Encode text chunks to video following memvid best practices."""
        try:
            # Initialize encoder with memvid's proven pattern
            encoder = self._create_memvid_encoder()
            
            # Add chunks
            encoder.add_chunks(chunks)
            logger.info(f"Added {len(chunks)} chunks to encoder")
            
            # Define paths
            video_path = str(self.output_dir / f"{episode_id}.mp4")
            index_path = str(self.output_dir / f"{episode_id}_index.json")
            
            # Build video with codec parameter (only supported parameter)
            codec = self.encoding_params['video_codec']
            logger.info(f"Building video with codec: {codec}")
            
            build_result = encoder.build_video(
                video_path, 
                index_path, 
                codec=codec,
                show_progress=True
            )
            
            # Get file statistics
            video_size = os.path.getsize(video_path) if os.path.exists(video_path) else 0
            text_size = sum(len(chunk.encode('utf-8')) for chunk in chunks)
            
            result = {
                "video_path": video_path,
                "index_path": index_path,
                "chunk_count": len(chunks),
                "video_size_bytes": video_size,
                "text_size_bytes": text_size,
                "encoding_params": self.encoding_params.copy(),
                "build_result": build_result
            }
            
            logger.info(f"Successfully encoded {len(chunks)} chunks to {video_path}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to encode chunks for episode {episode_id}: {str(e)}")
            raise
    
    def decode(self, video_path: str, index_path: str) -> List[str]:
        """Decode video back to text chunks following memvid best practices."""
        try:
            logger.info(f"Decoding video: {video_path}")
            
            # Validate files exist
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"Index file not found: {index_path}")
            
            # Initialize retriever (memvid handles embedding model automatically)
            retriever = MemvidRetriever(video_path, index_path)
            
            # Get all chunks by searching with broad query
            # Use empty string or broad search term to retrieve all chunks
            results = retriever.search("", top_k=10000)  # High top_k to get all chunks
            
            # Handle different result formats
            if isinstance(results, list):
                # Filter out empty results and extract text content
                chunks = []
                for result in results:
                    if isinstance(result, dict) and 'text' in result:
                        chunks.append(result['text'])
                    elif isinstance(result, str) and result.strip():
                        chunks.append(result)
                return chunks
            elif isinstance(results, str):
                return [results] if results.strip() else []
            else:
                logger.warning(f"Unexpected result type: {type(results)}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to decode video {video_path}: {str(e)}")
            raise
    
    def get_video_info(self, video_path: str, index_path: str) -> Dict[str, Any]:
        """Get information about a video file."""
        try:
            info = {
                "video_path": video_path,
                "index_path": index_path,
                "video_exists": os.path.exists(video_path),
                "index_exists": os.path.exists(index_path),
                "video_size_bytes": os.path.getsize(video_path) if os.path.exists(video_path) else 0,
                "encoding_params": self.encoding_params.copy()
            }
            
            # Try to get chunk count from index if available
            if os.path.exists(index_path):
                try:
                    import json
                    with open(index_path, 'r') as f:
                        index_data = json.load(f)
                    info["chunk_count"] = len(index_data.get('chunks', []))
                except Exception as e:
                    logger.warning(f"Could not read index file {index_path}: {e}")
                    info["chunk_count"] = "unknown"
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get video info for {video_path}: {str(e)}")
            raise