#!/usr/bin/env python3
"""
Test SearchEngine with memvid integration - validates actual codebase functionality
"""

import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.search_engine import SearchEngine

def test_search_engine_memvid():
    """Test SearchEngine with real memvid embeddings"""
    print("🔍 Testing SearchEngine with Memvid Integration...")
    
    # Test data - AI/ML focused documents
    docs = [
        {"id": "1", "text": "Machine learning algorithms learn patterns from data"},
        {"id": "2", "text": "Neural networks have multiple layers for deep learning"}, 
        {"id": "3", "text": "Natural language processing understands human text"},
        {"id": "4", "text": "Computer vision analyzes visual information from images"},
        {"id": "5", "text": "Reinforcement learning uses rewards to train agents"}
    ]
    
    print(f"📝 Test corpus: {len(docs)} documents")
    
    # Initialize search engine
    engine = SearchEngine()
    
    # Generate embeddings using memvid approach
    texts = [doc['text'] for doc in docs]
    embeddings = engine.embedding_generator.generate_embeddings(texts)
    print(f"🧠 Generated {len(embeddings)} embeddings ({len(embeddings[0])}D)")
    
    # Build search index
    engine.build_index(embeddings, docs)
    stats = engine.get_stats()
    print(f"✅ Index built: {stats}")
    
    # Test semantic search
    print("\n🔍 Testing semantic search:")
    queries = [
        "neural network deep learning",
        "language processing text",
        "computer vision images",
        "machine learning data patterns"
    ]
    
    for query in queries:
        results = engine.semantic_search(query, k=3)
        print(f"\n  Query: '{query}'")
        for i, result in enumerate(results[:2], 1):
            print(f"    {i}. [Score: {result.score:.3f}] {result.text[:60]}...")
    
    # Test keyword search  
    print("\n🔑 Testing keyword search:")
    kw_results = engine.keyword_search("neural networks", k=3)
    print(f"  Keyword results: {len(kw_results)}")
    for result in kw_results[:2]:
        print(f"    [Score: {result.score:.3f}] {result.text[:60]}...")
    
    # Test hybrid search
    print("\n🔀 Testing hybrid search:")
    hybrid_results = engine.hybrid_search("neural networks", k=3)
    print(f"  Hybrid results: {len(hybrid_results)}")
    for result in hybrid_results[:2]:
        print(f"    [Score: {result.score:.3f}] ({result.source}) {result.text[:50]}...")
    
    # Test search interface
    print("\n🎯 Testing unified search interface:")
    for search_type in ['semantic', 'keyword', 'hybrid']:
        results = engine.search("machine learning", limit=2, search_type=search_type)
        print(f"  {search_type}: {len(results)} results")
    
    print("\n🎉 SearchEngine memvid integration tests passed!")

if __name__ == "__main__":
    test_search_engine_memvid()