#!/usr/bin/env python3
"""
Core data models for podcast episodes and transcript chunks with memvid integration
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
from memvid import MemvidEncoder, MemvidChat, MemvidRetriever, chat_with_memory


@dataclass
class Chunk:
    """Transcript chunk with speaker information and timing"""
    id: str
    episode_id: str
    start_time: float
    end_time: float
    speaker: str
    text: str
    quality_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_to_encoder(self, encoder: MemvidEncoder) -> None:
        """Add this chunk to memvid encoder"""
        # Format text with metadata prefix for searchability
        formatted_text = f"[{self.speaker}] {self.text}"
        encoder.add_text(formatted_text)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "episode_id": self.episode_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "speaker": self.speaker,
            "text": self.text,
            "quality_score": self.quality_score,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chunk':
        """Create chunk from dictionary"""
        return cls(**data)


@dataclass
class Episode:
    """Podcast episode with metadata and chunks"""
    id: str
    title: str
    podcast_name: str
    date: datetime
    duration: float
    speakers: List[str]
    chunks: List[Chunk] = field(default_factory=list)
    video_file: Optional[str] = None
    chunk_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.chunk_count = len(self.chunks)
    
    def to_memvid_encoder(self) -> MemvidEncoder:
        """Convert episode to memvid encoder for video memory creation"""
        encoder = MemvidEncoder()
        for chunk in self.chunks:
            chunk.add_to_encoder(encoder)
        return encoder
    
    def create_chat_session(self, api_key: Optional[str] = None) -> MemvidChat:
        """Create chat session for this episode using memvid"""
        if not self.video_file:
            raise ValueError("Video file not set. Call build_video() first.")
        index_file = self.video_file.replace('.mp4', '_index.json')
        return MemvidChat(self.video_file, index_file, llm_api_key=api_key)
    
    def build_video(self, output_path: str) -> None:
        """Build memvid video from episode chunks"""
        encoder = self.to_memvid_encoder()
        index_path = output_path.replace('.mp4', '_index.json')
        encoder.build_video(output_path, index_path)
        self.video_file = output_path
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert episode to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "title": self.title,
            "podcast_name": self.podcast_name,
            "date": self.date.isoformat(),
            "duration": self.duration,
            "speakers": self.speakers,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "video_file": self.video_file,
            "chunk_count": self.chunk_count,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Episode':
        """Create episode from dictionary"""
        data["date"] = datetime.fromisoformat(data["date"])
        data["chunks"] = [Chunk.from_dict(chunk_data) for chunk_data in data["chunks"]]
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert episode to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Episode':
        """Create episode from JSON string"""
        return cls.from_dict(json.loads(json_str))
    
    def get_retriever(self) -> MemvidRetriever:
        """Get memvid retriever for semantic search"""
        if not self.video_file:
            raise ValueError("Video file not set. Call build_video() first.")
        index_file = self.video_file.replace('.mp4', '_index.json')
        return MemvidRetriever(self.video_file, index_file)
    
    def search_by_speaker(self, speaker: str, query: str = "", top_k: int = 5):
        """Search chunks by specific speaker"""
        retriever = self.get_retriever()
        results = retriever.search(f"speaker:{speaker} {query}".strip(), top_k=top_k)
        return results
    
    def search_by_timerange(self, start_time: float, end_time: float, query: str = "", top_k: int = 5):
        """Search chunks within specific time range"""
        retriever = self.get_retriever()
        time_query = f"time:{start_time}-{end_time} {query}".strip()
        return retriever.search(time_query, top_k=top_k) 