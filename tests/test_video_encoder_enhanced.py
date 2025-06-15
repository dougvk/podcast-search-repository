#!/usr/bin/env python3
"""Enhanced test script for memvid integration with best practices."""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.video_encoder import VideoEncoder

def test_memvid_best_practices():
    """Test memvid integration following best practices."""
    print("🧪 Testing memvid integration with best practices...")
    
    # Sample text chunks of varying sizes
    chunks = [
        "This is the first chunk of podcast transcript discussing artificial intelligence.",
        "Here we have a longer chunk where the guest explains their journey into machine learning, starting from their computer science background at university and moving through various roles in tech companies before focusing on AI research.",
        "Short chunk.",
        "The guest mentions their experience with neural networks and deep learning architectures, particularly focusing on transformer models and how they revolutionized natural language processing tasks.",
        "Another medium-length chunk covering the discussion about the future of AI and its potential impact on various industries including healthcare, finance, and education.",
        "Final chunk wrapping up the conversation with thoughts on ethical AI development."
    ]
    
    print(f"📝 Test chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"  {i+1}. {chunk[:60]}... ({len(chunk)} chars)")
    
    # Test different encoder configurations
    test_configs = [
        {
            "name": "Default Config",
            "params": {}
        },
        {
            "name": "H264 Codec",
            "params": {"video_codec": "h264"}
        },
        {
            "name": "H265 Codec", 
            "params": {"video_codec": "h265"}
        },
        {
            "name": "Custom Embedding Model",
            "params": {"embedding_model": "all-MiniLM-L6-v2"}
        }
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n🎬 Testing {config['name']}...")
        
        try:
            # Initialize encoder with test configuration
            encoder = VideoEncoder(**config['params'])
            
            # Encode chunks
            episode_id = f"test_{config['name'].lower().replace(' ', '_')}"
            result = encoder.encode(chunks, episode_id)
            
            print(f"✅ Video created: {result['video_path']}")
            print(f"📊 Index file: {result['index_path']}")
            print(f"📦 Chunks encoded: {result['chunk_count']}")
            print(f"💾 Text size: {result['text_size_bytes']} bytes")
            print(f"🎥 Video size: {result['video_size_bytes']} bytes")
            
            # Test decoding
            print("🔍 Testing decode...")
            decoded_chunks = encoder.decode(result['video_path'], result['index_path'])
            
            # Verify integrity
            encode_success = result['chunk_count'] == len(chunks)
            decode_success = len(decoded_chunks) == len(chunks)
            overall_success = encode_success and decode_success
            
            test_result = {
                "config": config['name'],
                "encode_success": encode_success,
                "decode_success": decode_success,
                "overall_success": overall_success,
                "original_chunks": len(chunks),
                "decoded_chunks": len(decoded_chunks),
                "text_size": result['text_size_bytes'],
                "video_size": result['video_size_bytes'],
                "encoding_params": result.get('encoding_params', {}),
                "build_result": result.get('build_result', {})
            }
            
            results.append(test_result)
            
            status = "✅ PASSED" if overall_success else "❌ FAILED"
            print(f"{status} - {config['name']}")
            
            # Test get_video_info method
            print("📋 Testing get_video_info...")
            info = encoder.get_video_info(result['video_path'], result['index_path'])
            print(f"   Video exists: {info['video_exists']}")
            print(f"   Index exists: {info['index_exists']}")
            print(f"   Chunk count: {info.get('chunk_count', 'unknown')}")
            
        except Exception as e:
            print(f"❌ FAILED - {config['name']}: {str(e)}")
            results.append({
                "config": config['name'],
                "encode_success": False,
                "decode_success": False,
                "overall_success": False,
                "error": str(e)
            })
    
    # Summary
    print("\n" + "="*60)
    print("📊 MEMVID INTEGRATION TEST SUMMARY")
    print("="*60)
    
    passed_tests = sum(1 for r in results if r.get('overall_success', False))
    total_tests = len(results)
    
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    for result in results:
        status = "✅" if result.get('overall_success', False) else "❌"
        config_name = result['config']
        if 'error' in result:
            print(f"{status} {config_name}: {result['error']}")
        else:
            print(f"{status} {config_name}: {result['original_chunks']} → {result['decoded_chunks']} chunks")
    
    print("\n🚀 Memvid best practices implementation complete!")
    return passed_tests == total_tests

if __name__ == "__main__":
    success = test_memvid_best_practices()
    sys.exit(0 if success else 1)