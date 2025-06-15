"""Video storage management with robust encode/decode pipeline."""
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import time
import hashlib
import logging
from datetime import datetime

from .video_encoder import VideoEncoder

# Memvid integration for efficient storage and retrieval
try:
    from memvid import MemvidEncoder, MemvidRetriever
    MEMVID_AVAILABLE = True
except ImportError:
    MEMVID_AVAILABLE = False
    logging.warning("Memvid not available. Using fallback storage.")

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    """Data structure for text chunks with metadata."""
    id: str
    episode_id: str
    text: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    speaker: Optional[str] = None
    quality_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate chunk data after initialization."""
        if not self.id:
            raise ValueError("Chunk ID cannot be empty")
        if not self.episode_id:
            raise ValueError("Episode ID cannot be empty")
        if not self.text or not self.text.strip():
            raise ValueError("Chunk text cannot be empty")

@dataclass
class StorageResult:
    """Result of storage operation with performance metrics."""
    success: bool
    video_path: str
    index_path: str
    chunk_count: int
    processing_time: float
    video_size_bytes: int
    text_size_bytes: int
    checksum: str
    error_message: Optional[str] = None
    
class MetadataIndexer:
    """Efficient metadata indexing with search capabilities."""
    
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.index_file = storage_dir / "metadata_index.json"
        self.search_index = {}
        self._load_index()
    
    def _load_index(self):
        """Load search index from disk."""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                self.search_index = json.load(f)
    
    def _save_index(self):
        """Save search index to disk."""
        with open(self.index_file, 'w') as f:
            json.dump(self.search_index, f, indent=2)
    
    def index_episode(self, episode_id: str, metadata: Dict[str, Any]):
        """Index episode metadata for fast search."""
        # Create searchable keywords
        keywords = set()
        keywords.add(episode_id.lower())
        if 'chunks' in metadata:
            for chunk in metadata['chunks']:
                if chunk.get('speaker'):
                    keywords.add(chunk['speaker'].lower())
                # Add first few words from text for content search
                text_words = chunk.get('text', '')[:100].lower().split()[:10]
                keywords.update(text_words)
        
        self.search_index[episode_id] = {
            'keywords': list(keywords),
            'metadata': metadata,
            'indexed_at': datetime.now().isoformat()
        }
        self._save_index()
    
    def search(self, query: str, limit: int = 10) -> List[str]:
        """Search episodes by query."""
        query_words = set(query.lower().split())
        scores = {}
        
        for episode_id, data in self.search_index.items():
            keywords = set(data['keywords'])
            score = len(query_words.intersection(keywords)) / len(query_words)
            if score > 0:
                scores[episode_id] = score
        
        return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:limit]
    
class StorageManager:
    """Robust storage manager with comprehensive encode/decode pipeline."""
    
    def __init__(self, 
                 storage_dir: str = "data/video_libraries",
                 video_encoder_config: Optional[Dict[str, Any]] = None,
                 use_memvid: bool = True):
        """Initialize StorageManager with error handling and validation."""
        try:
            self.storage_dir = Path(storage_dir)
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            
            # Create hierarchical directory structure
            self.episodes_dir = self.storage_dir / "episodes"
            self.archive_dir = self.storage_dir / "archive"
            self.temp_dir = self.storage_dir / "temp"
            
            for dir_path in [self.episodes_dir, self.archive_dir, self.temp_dir]:
                dir_path.mkdir(exist_ok=True)
            
            # Initialize video encoder with configuration
            encoder_config = video_encoder_config or {}
            encoder_config['output_dir'] = str(self.episodes_dir)
            self.video_encoder = VideoEncoder(**encoder_config)
            
            # Initialize memvid if available
            self.use_memvid = use_memvid and MEMVID_AVAILABLE
            if self.use_memvid:
                self.memvid_encoder = MemvidEncoder()
                logger.info("Memvid integration enabled")
            
            # Initialize metadata tracking and indexing
            self.metadata_file = self.storage_dir / "storage_metadata.json"
            self.indexer = MetadataIndexer(self.storage_dir)
            self._load_metadata()
            
            logger.info(f"StorageManager initialized with directory: {self.storage_dir}")
            
        except Exception as e:
            logger.error(f"Failed to initialize StorageManager: {str(e)}")
            raise
    
    def _get_episode_path(self, episode_id: str) -> Path:
        """Get hierarchical path for episode based on date/ID."""
        # Create year/month structure for organization
        date_str = datetime.now().strftime("%Y/%m")
        episode_dir = self.episodes_dir / date_str
        episode_dir.mkdir(parents=True, exist_ok=True)
        return episode_dir / f"{episode_id}.mp4"
    
    def _load_metadata(self):
        """Load metadata from storage."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {
                    "created": datetime.now().isoformat(),
                    "episodes": {},
                    "statistics": {
                        "total_episodes": 0,
                        "total_chunks": 0,
                        "total_storage_bytes": 0
                    }
                }
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}. Creating new metadata.")
            self.metadata = {
                "created": datetime.now().isoformat(),
                "episodes": {},
                "statistics": {
                    "total_episodes": 0,
                    "total_chunks": 0,
                    "total_storage_bytes": 0
                }
            }
    
    def _save_metadata(self):
        """Save metadata to storage."""
        try:
            self.metadata["last_updated"] = datetime.now().isoformat()
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _calculate_checksum(self, chunks: List[Chunk]) -> str:
        """Calculate checksum for data integrity verification."""
        try:
            # Create a deterministic representation of chunks
            chunk_data = []
            for chunk in sorted(chunks, key=lambda x: x.id):
                chunk_dict = asdict(chunk)
                chunk_data.append(json.dumps(chunk_dict, sort_keys=True))
            
            combined_data = "".join(chunk_data)
            return hashlib.sha256(combined_data.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate checksum: {e}")
            return ""
    
    def _validate_chunks(self, chunks: List[Chunk]) -> List[str]:
        """Validate chunks and return list of validation errors."""
        errors = []
        
        try:
            if not chunks:
                errors.append("No chunks provided")
                return errors
            
            chunk_ids = set()
            for i, chunk in enumerate(chunks):
                try:
                    # Validate chunk structure
                    if not isinstance(chunk, Chunk):
                        errors.append(f"Chunk {i} is not a valid Chunk object")
                        continue
                    
                    # Check for duplicate IDs
                    if chunk.id in chunk_ids:
                        errors.append(f"Duplicate chunk ID: {chunk.id}")
                    chunk_ids.add(chunk.id)
                    
                    # Validate text content
                    if not chunk.text or len(chunk.text.strip()) == 0:
                        errors.append(f"Chunk {chunk.id} has empty text")
                    
                    # Validate text length (reasonable limits)
                    if len(chunk.text) > 50000:  # 50KB limit per chunk
                        errors.append(f"Chunk {chunk.id} text too long ({len(chunk.text)} chars)")
                    
                    # Validate time ranges if provided
                    if chunk.start_time is not None and chunk.end_time is not None:
                        if chunk.start_time >= chunk.end_time:
                            errors.append(f"Chunk {chunk.id} has invalid time range")
                    
                except Exception as e:
                    errors.append(f"Error validating chunk {i}: {str(e)}")
            
        except Exception as e:
            errors.append(f"Critical validation error: {str(e)}")
        
        return errors
    
    def encode_chunks_to_video(self, chunks: List[Chunk]) -> StorageResult:
        """Encode chunks to video with memvid integration and validation."""
        start_time = time.time()
        
        try:
            logger.info(f"Starting encode operation for {len(chunks)} chunks")
            
            # Input validation
            validation_errors = self._validate_chunks(chunks)
            if validation_errors:
                error_msg = f"Validation failed: {'; '.join(validation_errors)}"
                logger.error(error_msg)
                return StorageResult(
                    success=False,
                    video_path="",
                    index_path="",
                    chunk_count=0,
                    processing_time=time.time() - start_time,
                    video_size_bytes=0,
                    text_size_bytes=0,
                    checksum="",
                    error_message=error_msg
                )
            
            # Extract episode ID
            episode_id = chunks[0].episode_id
            
            # Verify all chunks belong to same episode
            if not all(chunk.episode_id == episode_id for chunk in chunks):
                error_msg = "All chunks must belong to the same episode"
                logger.error(error_msg)
                return StorageResult(
                    success=False,
                    video_path="",
                    index_path="",
                    chunk_count=0,
                    processing_time=time.time() - start_time,
                    video_size_bytes=0,
                    text_size_bytes=0,
                    checksum="",
                    error_message=error_msg
                )
            
            # Use hierarchical path organization  
            video_path = str(self._get_episode_path(episode_id))
            index_path = str(Path(video_path).with_suffix('_index.json'))
            
            # Calculate metrics
            text_chunks = [chunk.text for chunk in chunks]
            text_size = sum(len(text.encode('utf-8')) for text in text_chunks)
            checksum = self._calculate_checksum(chunks)
            
            # Use memvid if available, fallback to video encoder
            try:
                if self.use_memvid:
                    # Memvid encoding - optimized for compression
                    self.memvid_encoder.add_chunks(text_chunks)
                    self.memvid_encoder.build_video(video_path, index_path, fps=30, frame_size=512)
                    success = Path(video_path).exists() and Path(index_path).exists()
                else:
                    # Fallback encoding
                    encode_result = self.video_encoder.encode(text_chunks, episode_id)
                    video_path = encode_result['video_path']
                    index_path = encode_result['index_path']
                    success = Path(video_path).exists()
                
                if not success:
                    raise Exception("Output files not created")
                    
            except Exception as e:
                error_msg = f"Encoding failed: {str(e)}"
                logger.error(error_msg)
                return StorageResult(
                    success=False, video_path=video_path, index_path=index_path,
                    chunk_count=len(chunks), processing_time=time.time() - start_time,
                    video_size_bytes=0, text_size_bytes=text_size,
                    checksum=checksum, error_message=error_msg
                )
            
            # Update metadata and index
            video_size = Path(video_path).stat().st_size
            metadata = {
                'episode_id': episode_id, 'video_path': video_path, 'index_path': index_path,
                'chunk_count': len(chunks), 'checksum': checksum, 'chunks': chunks,
                'created': datetime.now().isoformat(), 'video_size_bytes': video_size
            }
            self._update_episode_metadata(episode_id, chunks, video_path, index_path, checksum)
            self.indexer.index_episode(episode_id, metadata)
            
            processing_time = time.time() - start_time
            
            result = StorageResult(
                success=True,
                video_path=video_path,
                index_path=index_path,
                chunk_count=len(chunks),
                processing_time=processing_time,
                video_size_bytes=video_size,
                text_size_bytes=text_size,
                checksum=checksum
            )
            
            logger.info(f"Successfully encoded {len(chunks)} chunks in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            error_msg = f"Critical error during encoding: {str(e)}"
            logger.error(error_msg)
            return StorageResult(
                success=False,
                video_path="",
                index_path="",
                chunk_count=len(chunks) if chunks else 0,
                processing_time=time.time() - start_time,
                video_size_bytes=0,
                text_size_bytes=0,
                checksum="",
                error_message=error_msg
            )
    
    def decode_chunks_from_video(self, video_path: str) -> List[Chunk]:
        """Decode chunks from video with memvid integration."""
        try:
            logger.info(f"Decoding video: {video_path}")
            
            video_file = Path(video_path)
            if not video_file.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            index_path = str(video_file.with_suffix('_index.json'))
            if not Path(index_path).exists():
                raise FileNotFoundError(f"Index file not found: {index_path}")
            
            # Use memvid if available
            if self.use_memvid:
                retriever = MemvidRetriever(video_path, index_path)
                # Get all chunks from the video - using a broad search to get everything
                search_results = retriever.search("", top_k=10000)  # Large number to get all
                text_chunks = [chunk for chunk, score in search_results]
            else:
                text_chunks = self.video_encoder.decode(video_path, index_path)
            
            # Reconstruct Chunk objects from metadata
            episode_id = video_file.stem.replace('_index', '')
            episode_metadata = self.metadata.get("episodes", {}).get(episode_id, {})
            chunk_metadata = episode_metadata.get("chunks", [])
            
            # Reconstruct Chunk objects
            chunks = []
            for i, text in enumerate(text_chunks):
                # Use stored metadata if available, otherwise create basic chunk
                if i < len(chunk_metadata):
                    metadata = chunk_metadata[i]
                    chunk = Chunk(
                        id=metadata.get("id", f"{episode_id}_chunk_{i}"),
                        episode_id=episode_id,
                        text=text,
                        start_time=metadata.get("start_time"),
                        end_time=metadata.get("end_time"),
                        speaker=metadata.get("speaker"),
                        quality_score=metadata.get("quality_score"),
                        metadata=metadata.get("metadata")
                    )
                else:
                    chunk = Chunk(
                        id=f"{episode_id}_chunk_{i}",
                        episode_id=episode_id,
                        text=text
                    )
                chunks.append(chunk)
            
            # Verify integrity if checksum available
            stored_checksum = episode_metadata.get("checksum")
            if stored_checksum:
                calculated_checksum = self._calculate_checksum(chunks)
                if stored_checksum != calculated_checksum:
                    logger.warning(f"Checksum mismatch for {episode_id}. Data may be corrupted.")
            
            logger.info(f"Successfully decoded {len(chunks)} chunks from {video_path}")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to decode video {video_path}: {str(e)}")
            raise
    
    def _update_episode_metadata(self, episode_id: str, chunks: List[Chunk], 
                                video_path: str, index_path: str, checksum: str):
        """Update metadata for an episode."""
        try:
            chunk_metadata = []
            for chunk in chunks:
                chunk_meta = {
                    "id": chunk.id,
                    "start_time": chunk.start_time,
                    "end_time": chunk.end_time,
                    "speaker": chunk.speaker,
                    "quality_score": chunk.quality_score,
                    "metadata": chunk.metadata,
                    "text_length": len(chunk.text)
                }
                chunk_metadata.append(chunk_meta)
            
            episode_metadata = {
                "episode_id": episode_id,
                "video_path": video_path,
                "index_path": index_path,
                "chunk_count": len(chunks),
                "checksum": checksum,
                "chunks": chunk_metadata,
                "created": datetime.now().isoformat(),
                "video_size_bytes": Path(video_path).stat().st_size if Path(video_path).exists() else 0
            }
            
            self.metadata["episodes"][episode_id] = episode_metadata
            
            # Update statistics
            stats = self.metadata["statistics"]
            stats["total_episodes"] = len(self.metadata["episodes"])
            stats["total_chunks"] = sum(ep.get("chunk_count", 0) for ep in self.metadata["episodes"].values())
            stats["total_storage_bytes"] = sum(ep.get("video_size_bytes", 0) for ep in self.metadata["episodes"].values())
            
            self._save_metadata()
            
        except Exception as e:
            logger.error(f"Failed to update episode metadata: {e}")
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Retrieve a specific chunk by ID."""
        try:
            # Search through all episodes for the chunk
            for episode_id, episode_data in self.metadata.get("episodes", {}).items():
                chunk_metadata = episode_data.get("chunks", [])
                for chunk_meta in chunk_metadata:
                    if chunk_meta.get("id") == chunk_id:
                        # Load the video and decode to get the actual text
                        video_path = episode_data.get("video_path")
                        if video_path and Path(video_path).exists():
                            chunks = self.decode_chunks_from_video(video_path)
                            # Find the matching chunk
                            for chunk in chunks:
                                if chunk.id == chunk_id:
                                    return chunk
            
            logger.warning(f"Chunk not found: {chunk_id}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve chunk {chunk_id}: {str(e)}")
            return None
    
    def organize_video_library(self) -> Dict[str, Any]:
        """Organize and validate the video library structure."""
        try:
            logger.info("Organizing video library...")
            
            organization_report = {
                "total_episodes": 0,
                "valid_episodes": 0,
                "corrupted_episodes": [],
                "missing_files": [],
                "orphaned_files": [],
                "total_size_bytes": 0
            }
            
            # Check all registered episodes
            for episode_id, episode_data in self.metadata.get("episodes", {}).items():
                organization_report["total_episodes"] += 1
                
                video_path = episode_data.get("video_path", "")
                index_path = episode_data.get("index_path", "")
                
                # Check if files exist
                video_exists = Path(video_path).exists() if video_path else False
                index_exists = Path(index_path).exists() if index_path else False
                
                if video_exists and index_exists:
                    # Verify integrity
                    try:
                        chunks = self.decode_chunks_from_video(video_path)
                        expected_count = episode_data.get("chunk_count", 0)
                        if len(chunks) == expected_count:
                            organization_report["valid_episodes"] += 1
                            organization_report["total_size_bytes"] += Path(video_path).stat().st_size
                        else:
                            organization_report["corrupted_episodes"].append({
                                "episode_id": episode_id,
                                "reason": f"Chunk count mismatch: expected {expected_count}, got {len(chunks)}"
                            })
                    except Exception as e:
                        organization_report["corrupted_episodes"].append({
                            "episode_id": episode_id,
                            "reason": f"Decode error: {str(e)}"
                        })
                else:
                    missing_files = []
                    if not video_exists:
                        missing_files.append(video_path)
                    if not index_exists:
                        missing_files.append(index_path)
                    
                    organization_report["missing_files"].append({
                        "episode_id": episode_id,
                        "missing_files": missing_files
                    })
            
            # Check for orphaned files in storage directory
            video_files = list(self.storage_dir.glob("*.mp4"))
            index_files = list(self.storage_dir.glob("*_index.json"))
            
            registered_videos = set()
            registered_indices = set()
            
            for episode_data in self.metadata.get("episodes", {}).values():
                if episode_data.get("video_path"):
                    registered_videos.add(Path(episode_data["video_path"]).name)
                if episode_data.get("index_path"):
                    registered_indices.add(Path(episode_data["index_path"]).name)
            
            for video_file in video_files:
                if video_file.name not in registered_videos:
                    organization_report["orphaned_files"].append(str(video_file))
            
            for index_file in index_files:
                if index_file.name not in registered_indices:
                    organization_report["orphaned_files"].append(str(index_file))
            
            logger.info(f"Library organization complete: {organization_report['valid_episodes']}/{organization_report['total_episodes']} episodes valid")
            return organization_report
            
        except Exception as e:
            logger.error(f"Failed to organize video library: {str(e)}")
            raise
    
    def search_episodes(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search episodes using metadata indexer."""
        episode_ids = self.indexer.search(query, limit)
        results = []
        for episode_id in episode_ids:
            episode_data = self.metadata.get("episodes", {}).get(episode_id, {})
            if episode_data:
                results.append({
                    'episode_id': episode_id,
                    'chunk_count': episode_data.get('chunk_count', 0),
                    'created': episode_data.get('created', ''),
                    'video_size_bytes': episode_data.get('video_size_bytes', 0)
                })
        return results
    
    def archive_episode(self, episode_id: str) -> bool:
        """Move episode to archive directory for lifecycle management."""
        try:
            episode_data = self.metadata.get("episodes", {}).get(episode_id)
            if not episode_data:
                return False
            
            video_path = Path(episode_data['video_path'])
            index_path = Path(episode_data['index_path'])
            
            if video_path.exists():
                archive_video = self.archive_dir / video_path.name
                video_path.rename(archive_video)
                episode_data['video_path'] = str(archive_video)
                
            if index_path.exists():
                archive_index = self.archive_dir / index_path.name
                index_path.rename(archive_index)
                episode_data['index_path'] = str(archive_index)
                
            self._save_metadata()
            logger.info(f"Archived episode: {episode_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to archive episode {episode_id}: {e}")
            return False

class VideoStorage(StorageManager):
    """Legacy compatibility class - redirects to StorageManager."""
    
    def store(self, video_path: str, metadata: Dict[str, Any]) -> str:
        """Store video with metadata."""
        logger.warning("Using legacy VideoStorage.store() method. Consider using StorageManager.encode_chunks_to_video()")
        return video_path
    
    def retrieve(self, video_id: str) -> Dict[str, Any]:
        """Retrieve video and metadata."""
        logger.warning("Using legacy VideoStorage.retrieve() method. Consider using StorageManager.decode_chunks_from_video()")
        return {}