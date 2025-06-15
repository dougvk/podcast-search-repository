#!/usr/bin/env python3
"""Test script for StorageManager encode/decode pipeline."""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.storage import StorageManager, Chunk, StorageResult

def test_storage_manager_pipeline():
    """Test the complete StorageManager encode/decode pipeline."""
    print("🧪 Testing StorageManager encode/decode pipeline...")
    
    # Create test chunks with rich metadata
    test_chunks = [
        Chunk(
            id="ep001_chunk_001",
            episode_id="episode_001",
            text="Welcome to our podcast about artificial intelligence and machine learning.",
            start_time=0.0,
            end_time=5.2,
            speaker="Host",
            quality_score=0.95,
            metadata={"topic": "introduction", "confidence": 0.98}
        ),
        Chunk(
            id="ep001_chunk_002",
            episode_id="episode_001",
            text="Today we're joined by Dr. Sarah Chen, a leading researcher in deep learning architectures who has published over 50 papers in top-tier conferences.",
            start_time=5.2,
            end_time=12.8,
            speaker="Host",
            quality_score=0.92,
            metadata={"topic": "guest_introduction", "confidence": 0.95}
        ),
        Chunk(
            id="ep001_chunk_003",
            episode_id="episode_001",
            text="Thank you for having me. I'm excited to discuss the latest developments in transformer models and their applications.",
            start_time=12.8,
            end_time=19.5,
            speaker="Dr. Sarah Chen",
            quality_score=0.88,
            metadata={"topic": "guest_response", "confidence": 0.91}
        ),
        Chunk(
            id="ep001_chunk_004",
            episode_id="episode_001",
            text="Let's start with the fundamentals. Can you explain how attention mechanisms work in transformer architectures?",
            start_time=19.5,
            end_time=26.1,
            speaker="Host",
            quality_score=0.94,
            metadata={"topic": "technical_question", "confidence": 0.97}
        )
    ]
    
    print(f"📝 Created {len(test_chunks)} test chunks")
    for chunk in test_chunks:
        print(f"  • {chunk.id}: {chunk.text[:50]}... ({chunk.speaker})")
    
    # Initialize StorageManager
    storage_manager = StorageManager(
        storage_dir="data/test_storage",
        video_encoder_config={
            "video_codec": "h264",
            "embedding_model": "all-MiniLM-L6-v2"
        }
    )
    
    print(f"\n🏗️  StorageManager initialized in: {storage_manager.storage_dir}")
    
    # Test 1: Encode chunks to video
    print("\n🎬 Test 1: Encoding chunks to video...")
    
    encode_result = storage_manager.encode_chunks_to_video(test_chunks)
    
    if encode_result.success:
        print(f"✅ Encoding successful!")
        print(f"   Video: {encode_result.video_path}")
        print(f"   Index: {encode_result.index_path}")
        print(f"   Chunks: {encode_result.chunk_count}")
        print(f"   Processing time: {encode_result.processing_time:.2f}s")
        print(f"   Text size: {encode_result.text_size_bytes} bytes")
        print(f"   Video size: {encode_result.video_size_bytes} bytes")
        print(f"   Checksum: {encode_result.checksum[:16]}...")
    else:
        print(f"❌ Encoding failed: {encode_result.error_message}")
        return False
    
    # Test 2: Decode chunks from video
    print("\n🔍 Test 2: Decoding chunks from video...")
    
    try:
        decoded_chunks = storage_manager.decode_chunks_from_video(encode_result.video_path)
        print(f"✅ Decoding successful!")
        print(f"   Decoded {len(decoded_chunks)} chunks")
        
        # Verify data integrity
        integrity_check = True
        if len(decoded_chunks) != len(test_chunks):
            print(f"❌ Chunk count mismatch: expected {len(test_chunks)}, got {len(decoded_chunks)}")
            integrity_check = False
        
        for i, (original, decoded) in enumerate(zip(test_chunks, decoded_chunks)):
            if original.text != decoded.text:
                print(f"❌ Text mismatch in chunk {i}")
                integrity_check = False
            if original.id != decoded.id:
                print(f"❌ ID mismatch in chunk {i}")
                integrity_check = False
            if original.speaker != decoded.speaker:
                print(f"❌ Speaker mismatch in chunk {i}")
                integrity_check = False
        
        if integrity_check:
            print("✅ Data integrity verified!")
        else:
            print("❌ Data integrity check failed!")
            return False
            
    except Exception as e:
        print(f"❌ Decoding failed: {str(e)}")
        return False
    
    # Test 3: Get chunk by ID
    print("\n🔎 Test 3: Retrieving chunk by ID...")
    
    test_chunk_id = test_chunks[1].id
    retrieved_chunk = storage_manager.get_chunk_by_id(test_chunk_id)
    
    if retrieved_chunk:
        print(f"✅ Chunk retrieved successfully!")
        print(f"   ID: {retrieved_chunk.id}")
        print(f"   Text: {retrieved_chunk.text[:50]}...")
        print(f"   Speaker: {retrieved_chunk.speaker}")
        
        if retrieved_chunk.text == test_chunks[1].text:
            print("✅ Retrieved chunk matches original!")
        else:
            print("❌ Retrieved chunk doesn't match original!")
            return False
    else:
        print(f"❌ Failed to retrieve chunk: {test_chunk_id}")
        return False
    
    # Test 4: Organize video library
    print("\n📋 Test 4: Organizing video library...")
    
    try:
        organization_report = storage_manager.organize_video_library()
        print(f"✅ Library organization complete!")
        print(f"   Total episodes: {organization_report['total_episodes']}")
        print(f"   Valid episodes: {organization_report['valid_episodes']}")
        print(f"   Total size: {organization_report['total_size_bytes']} bytes")
        
        if organization_report['corrupted_episodes']:
            print(f"   ⚠️  Corrupted episodes: {len(organization_report['corrupted_episodes'])}")
        if organization_report['missing_files']:
            print(f"   ⚠️  Missing files: {len(organization_report['missing_files'])}")
        if organization_report['orphaned_files']:
            print(f"   ⚠️  Orphaned files: {len(organization_report['orphaned_files'])}")
            
    except Exception as e:
        print(f"❌ Library organization failed: {str(e)}")
        return False
    
    # Test 5: Error handling validation
    print("\n🚫 Test 5: Error handling validation...")
    
    # Test with invalid chunks (empty ID should be caught during construction)
    try:
        invalid_chunk = Chunk(id="", episode_id="test", text="Empty ID test")
        print("❌ Empty ID validation failed - should have raised ValueError")
        return False
    except ValueError as e:
        print("✅ Empty ID validation working!")
        print(f"   Error: {str(e)}")
    
    # Test with empty text (should be caught during construction)
    try:
        invalid_chunk = Chunk(id="valid_id", episode_id="test", text="")
        print("❌ Empty text validation failed - should have raised ValueError")
        return False
    except ValueError as e:
        print("✅ Empty text validation working!")
        print(f"   Error: {str(e)}")
    
    # Test with empty list of chunks
    try:
        empty_result = storage_manager.encode_chunks_to_video([])
        if not empty_result.success and "No chunks provided" in empty_result.error_message:
            print("✅ Empty chunk list validation working!")
        else:
            print("❌ Empty chunk list validation failed")
            return False
    except Exception as e:
        print(f"❌ Unexpected error during empty list test: {str(e)}")
        return False
    
    # Test with mixed episode IDs
    mixed_chunks = [
        Chunk(id="chunk1", episode_id="episode1", text="First episode chunk"),
        Chunk(id="chunk2", episode_id="episode2", text="Second episode chunk"),
    ]
    
    mixed_result = storage_manager.encode_chunks_to_video(mixed_chunks)
    if not mixed_result.success and "same episode" in mixed_result.error_message:
        print("✅ Mixed episode ID validation working!")
    else:
        print("❌ Mixed episode ID validation failed")
        return False
    
    print("\n" + "="*60)
    print("📊 STORAGE MANAGER TEST SUMMARY")
    print("="*60)
    print("✅ All tests passed successfully!")
    print("🚀 StorageManager encode/decode pipeline is robust and ready!")
    
    return True

def test_performance_monitoring():
    """Test performance monitoring capabilities."""
    print("\n⚡ Performance Monitoring Test")
    print("-" * 40)
    
    # Create larger dataset for performance testing
    large_chunks = []
    for i in range(20):
        chunk = Chunk(
            id=f"perf_chunk_{i:03d}",
            episode_id="performance_test",
            text=f"This is performance test chunk number {i}. " * 10,  # Make text longer
            start_time=i * 5.0,
            end_time=(i + 1) * 5.0,
            speaker=f"Speaker_{i % 3}",
            quality_score=0.9 + (i % 10) * 0.01
        )
        large_chunks.append(chunk)
    
    storage_manager = StorageManager(storage_dir="data/perf_test")
    
    print(f"📊 Testing with {len(large_chunks)} chunks")
    
    # Encode and measure performance
    result = storage_manager.encode_chunks_to_video(large_chunks)
    
    if result.success:
        print(f"✅ Performance test completed!")
        print(f"   Processing time: {result.processing_time:.3f}s")
        print(f"   Throughput: {result.chunk_count / result.processing_time:.1f} chunks/second")
        print(f"   Compression efficiency: {result.text_size_bytes / result.video_size_bytes:.3f}")
        
        if result.processing_time < 30.0:  # Should complete within 30 seconds
            print("✅ Performance within acceptable limits!")
            return True
        else:
            print("⚠️  Performance slower than expected")
            return False
    else:
        print(f"❌ Performance test failed: {result.error_message}")
        return False

if __name__ == "__main__":
    print("🧪 Starting comprehensive StorageManager tests...\n")
    
    # Run main pipeline test
    pipeline_success = test_storage_manager_pipeline()
    
    # Run performance test
    performance_success = test_performance_monitoring()
    
    overall_success = pipeline_success and performance_success
    
    if overall_success:
        print("\n🎉 All StorageManager tests completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)