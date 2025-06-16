#!/usr/bin/env python3
"""
TranscriptProcessor - Simple TXT transcript processor for memvid integration
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from memvid import MemvidEncoder


class TranscriptProcessor:
    """Simple wrapper around memvid's text processing for TXT transcript files"""
    
    def __init__(self):
        self.content: Optional[str] = None
        self.file_path: Optional[str] = None
        self.stats: Dict[str, Any] = {}
    
    def parse_transcript(self, file_path: str) -> str:
        """Read and validate TXT transcript file"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Transcript file not found: {file_path}")
        if not path.suffix.lower() == '.txt':
            raise ValueError(f"Only TXT files supported, got: {path.suffix}")
        
        self.content = path.read_text(encoding='utf-8')
        self.file_path = file_path
        self._calculate_stats()
        return self.content
    
    def process_with_memvid(self, encoder: MemvidEncoder, chunk_size: int = 512, overlap: int = 50) -> None:
        """Add transcript content to memvid encoder"""
        if not self.content:
            raise ValueError("No content loaded. Call parse_transcript() first.")
        
        encoder.add_text(self.content, chunk_size=chunk_size, overlap=overlap)
        self.stats.update({'chunk_size': chunk_size, 'overlap': overlap})
    
    def get_basic_stats(self) -> Dict[str, Any]:
        """Return basic file statistics and metadata"""
        if not self.file_path:
            return {}
        return self.stats
    
    def _calculate_stats(self) -> None:
        """Calculate basic file statistics"""
        if not self.content or not self.file_path:
            return
        
        path = Path(self.file_path)
        self.stats = {
            'filename': path.name,
            'file_size_bytes': path.stat().st_size,
            'character_count': len(self.content),
            'word_count': len(self.content.split()),
            'line_count': len(self.content.splitlines())
        } 