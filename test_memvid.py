#!/usr/bin/env python3
"""Simple test script for memvid integration."""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.video_encoder import VideoEncoder

def test_memvid_integration():
    """Test basic encode/decode functionality."""
    print("🧪 Testing memvid integration...")
    
    # Sample text chunks
    chunks = [
        "This is the first chunk of podcast transcript.",
        "Here we discuss artificial intelligence and machine learning.",
        "The guest mentions their experience with neural networks.",
        "They explain how transformers revolutionized NLP."
    ]
    
    print(f"📝 Original chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"  {i+1}. {chunk[:50]}...")
    
    # Initialize encoder
    encoder = VideoEncoder()
    
    # Encode to video
    print("\n🎬 Encoding chunks to video...")
    result = encoder.encode(chunks, "test_episode")
    
    print(f"✅ Video created: {result['video_path']}")
    print(f"📊 Index file: {result['index_path']}")
    print(f"📦 Chunks encoded: {result['chunk_count']}")
    
    # Check file sizes
    video_size = os.path.getsize(result['video_path'])
    text_size = sum(len(chunk.encode('utf-8')) for chunk in chunks)
    compression_ratio = text_size / video_size if video_size > 0 else 0
    
    print(f"💾 Original text size: {text_size} bytes")
    print(f"🎥 Video file size: {video_size} bytes")
    print(f"🗜️  Compression ratio: {compression_ratio:.2f}:1")
    
    # Decode from video
    print("\n🔍 Decoding video back to chunks...")
    decoded_chunks = encoder.decode(result['video_path'], result['index_path'])
    
    print(f"✅ Decoded chunks: {len(decoded_chunks)}")
    
    # Verify integrity
    success = len(chunks) == len(decoded_chunks)
    print(f"\n{'✅' if success else '❌'} Test {'PASSED' if success else 'FAILED'}")
    
    if success:
        print("🚀 Memvid integration working perfectly!")
    else:
        print("⚠️  Chunk count mismatch - investigation needed")
    
    return success

if __name__ == "__main__":
    test_memvid_integration()