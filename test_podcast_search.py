#!/usr/bin/env python3

import sys
from pathlib import Path
from memvid import MemvidRetriever

def test_podcast_search():
    """Test searching the podcast batch dataset"""
    
    # Check if files exist
    video_file = Path("data/podcast_batch_001.mp4")
    index_file = Path("data/podcast_batch_001_index.json")
    
    if not video_file.exists():
        print(f"❌ Video file not found: {video_file}")
        return False
    
    if not index_file.exists():
        print(f"❌ Index file not found: {index_file}")
        return False
    
    print(f"✅ Found video file: {video_file} ({video_file.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"✅ Found index file: {index_file}")
    
    # Initialize retriever
    print("\n🔍 Initializing memvid retriever...")
    try:
        retriever = MemvidRetriever(str(video_file), str(index_file))
        print("✅ Retriever initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize retriever: {e}")
        return False
    
    # Test queries
    test_queries = [
        "World War",
        "Hitler",
        "Olympics",
        "Ireland",
        "Taj Mahal",
        "Boris Johnson",
        "medieval",
        "dynasty"
    ]
    
    print(f"\n🎯 Testing {len(test_queries)} search queries...")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        try:
            results = retriever.search(query, top_k=3)
            if results:
                print(f"   Found {len(results)} results:")
                for j, text in enumerate(results, 1):
                    # Truncate long text for display
                    display_text = text[:150] + "..." if len(text) > 150 else text
                    print(f"   {j}. {display_text}")
                    print("      " + "-" * 50)
            else:
                print("   No results found")
        except Exception as e:
            print(f"   ❌ Search failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Search test completed!")
    return True

def interactive_search():
    """Interactive search mode"""
    video_file = Path("data/podcast_batch_001.mp4")
    index_file = Path("data/podcast_batch_001_index.json")
    
    if not video_file.exists() or not index_file.exists():
        print("❌ Dataset files not found. Run test mode first.")
        return
    
    try:
        retriever = MemvidRetriever(str(video_file), str(index_file))
        print("🔍 Interactive search mode - Enter queries (type 'quit' to exit)")
        print("=" * 60)
        
        while True:
            query = input("\nEnter search query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            try:
                results = retriever.search(query, top_k=5)
                if results:
                    print(f"\nFound {len(results)} results for '{query}':")
                    for i, text in enumerate(results, 1):
                        print(f"\n{i}. {text}")
                        print("-" * 60)
                else:
                    print(f"No results found for '{query}'")
            except Exception as e:
                print(f"❌ Search failed: {e}")
    
    except Exception as e:
        print(f"❌ Failed to initialize retriever: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_search()
    else:
        test_podcast_search() 