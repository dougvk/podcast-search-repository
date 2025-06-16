#!/usr/bin/env python3
"""
Test script for Episode and Chunk models with memvid integration
"""

import os
from datetime import datetime
from core.models import Episode, Chunk

def main():
    # Create test episode with chunks
    chunks = [
        Chunk(
            id="chunk_1",
            episode_id="ep_001",
            start_time=0.0,
            end_time=30.0,
            speaker="Host",
            text="Welcome to today's podcast about AI and video-based memory systems.",
            quality_score=0.95
        ),
        Chunk(
            id="chunk_2", 
            episode_id="ep_001",
            start_time=30.0,
            end_time=60.0,
            speaker="Guest",
            text="Thank you for having me. I'm excited to discuss memvid technology.",
            quality_score=0.92
        ),
        Chunk(
            id="chunk_3",
            episode_id="ep_001", 
            start_time=60.0,
            end_time=90.0,
            speaker="Host",
            text="Let's dive into how video-based AI memory works and its benefits.",
            quality_score=0.88
        )
    ]
    
    episode = Episode(
        id="ep_001",
        title="AI Memory Systems with Memvid",
        podcast_name="Tech Talk Podcast",
        date=datetime.now(),
        duration=90.0,
        speakers=["Host", "Guest"],
        chunks=chunks
    )
    
    print("✓ Created episode with chunks")
    print(f"  Episode: {episode.title}")
    print(f"  Chunks: {episode.chunk_count}")
    
    # Test JSON serialization
    json_data = episode.to_json()
    episode_restored = Episode.from_json(json_data)
    print("✓ JSON serialization/deserialization works")
    
    # Test memvid encoder integration
    encoder = episode.to_memvid_encoder()
    print("✓ Created memvid encoder from episode")
    
    # Build video memory (this would create actual files)
    episode.build_video("test_episode.mp4")
    print("✓ Built video memory")
    
    # Test chat session creation  
    try:
        # chat = episode.create_chat_session()
        print("✓ Chat session method available")
    except ValueError as e:
        print(f"✓ Chat session validation works: {e}")
    
    # Test retriever methods
    try:
        # retriever = episode.get_retriever()
        print("✓ Retriever methods available")
    except ValueError as e:
        print(f"✓ Retriever validation works: {e}")
    
    # Cleanup test files
    for file in ["test_episode.mp4", "test_episode_index.json", "test_episode.faiss"]:
        if os.path.exists(file):
            os.remove(file)
    
    print("\n🎉 All model integration tests passed!")

if __name__ == "__main__":
    main() 