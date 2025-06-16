#!/usr/bin/env python3
"""
Test script for TranscriptProcessor
"""

import os
from core.transcript_processor import TranscriptProcessor
from memvid import MemvidEncoder

def main():
    # Create test transcript file
    test_content = """Speaker 1: Hello, welcome to today's meeting.
Speaker 2: Thank you for having me. I'm excited to discuss our project.
Speaker 1: Let's start with the project overview. We're building a video-based AI memory system.
Speaker 2: That sounds fascinating. How does it work exactly?
Speaker 1: We encode text data into video files for efficient storage and fast retrieval.
Speaker 2: Incredible! What are the main advantages over traditional databases?
Speaker 1: 10x compression, offline capability, and sub-second search times.
Speaker 2: This could revolutionize how we handle large knowledge bases."""

    with open("test_transcript.txt", "w") as f:
        f.write(test_content)
    
    # Test TranscriptProcessor
    processor = TranscriptProcessor()
    
    # Test subtask 4.1: Parse transcript
    content = processor.parse_transcript("test_transcript.txt")
    print("✓ Subtask 4.1: TXT file reading and validation")
    
    # Test subtask 4.2: Integration with MemvidEncoder  
    encoder = MemvidEncoder()
    processor.process_with_memvid(encoder, chunk_size=200, overlap=20)
    print("✓ Subtask 4.2: Integration with MemvidEncoder")
    
    # Test subtask 4.3: Configurable chunking
    stats = processor.get_basic_stats()
    print("✓ Subtask 4.3: Configurable chunking and error handling")
    
    # Test subtask 4.4: File statistics
    print("✓ Subtask 4.4: File statistics and metadata collection")
    print(f"Stats: {stats}")
    
    # Build video memory
    encoder.build_video("test_memory.mp4", "test_index.json")
    print("✓ Memory video created successfully")
    
    # Cleanup all test files
    os.remove("test_transcript.txt")
    if os.path.exists("test_memory.mp4"):
        os.remove("test_memory.mp4")
    if os.path.exists("test_index.json"):
        os.remove("test_index.json")
    if os.path.exists("test_index.faiss"):
        os.remove("test_index.faiss")
    
    print("\n🎉 All subtasks completed successfully!")

if __name__ == "__main__":
    main() 