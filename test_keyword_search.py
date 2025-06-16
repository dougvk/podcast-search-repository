#!/usr/bin/env python3
"""Test keyword search implementation."""

import numpy as np
from core.search_engine import SearchEngine, SearchResult

def test_keyword_search():
    """Test keyword search with TF-IDF scoring."""
    print("🔍 Testing Keyword Search...")
    
    # Test documents with varied vocabulary
    sample_docs = [
        {"id": "1", "text": "Machine learning algorithms learn patterns from data to make predictions"},
        {"id": "2", "text": "Natural language processing helps computers understand human language"},
        {"id": "3", "text": "Deep neural networks have multiple layers for complex pattern recognition"},
        {"id": "4", "text": "Computer vision enables machines to interpret and analyze visual information"},
        {"id": "5", "text": "Reinforcement learning agents learn through trial and error with rewards"}
    ]
    
    print(f"📝 Test corpus: {len(sample_docs)} documents")
    
    # Create simple embeddings (we need them for index building)
    np.random.seed(42)
    embeddings = []
    for i in range(len(sample_docs)):
        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding.tolist())
    
    # Build search engine
    engine = SearchEngine()
    engine.build_index(embeddings, sample_docs)
    
    print(f"  ✅ Index built: {engine.get_stats()}")
    
    # Test keyword searches
    print("\n🔍 Testing keyword searches:")
    
    test_queries = [
        "machine learning algorithms",
        "language processing",
        "neural networks layers",
        "computer vision",
        "reinforcement learning"
    ]
    
    for query in test_queries:
        results = engine.keyword_search(query, k=3)
        
        print(f"\n  Query: '{query}'")
        if results:
            for i, result in enumerate(results, 1):
                print(f"    {i}. [Score: {result.score:.3f}] {result.text[:60]}...")
        else:
            print("    No results found")
    
    # Test specific keyword matches
    print("\n🎯 Testing specific keyword matches:")
    
    # Single word that should match multiple docs
    learn_results = engine.keyword_search("learn", k=5)
    print(f"\n  Query: 'learn' - {len(learn_results)} results")
    for i, result in enumerate(learn_results, 1):
        print(f"    {i}. [Score: {result.score:.3f}] ...{result.text[:50]}...")
    
    # Exact phrase
    pattern_results = engine.keyword_search("pattern recognition", k=3)
    print(f"\n  Query: 'pattern recognition' - {len(pattern_results)} results")
    for i, result in enumerate(pattern_results, 1):
        print(f"    {i}. [Score: {result.score:.3f}] ...{result.text[:50]}...")
    
    # Test fuzzy search
    print("\n🔧 Testing fuzzy keyword search:")
    
    # Typo in "machine"
    typo_results = engine.fuzzy_keyword_search("machne learning", k=3)
    print(f"\n  Query: 'machne learning' (typo) - {len(typo_results)} results")
    for i, result in enumerate(typo_results, 1):
        print(f"    {i}. [Score: {result.score:.3f}] ...{result.text[:50]}...")
    
    # Test edge cases
    print("\n🧪 Testing edge cases:")
    
    # Empty query
    empty_results = engine.keyword_search("", k=5)
    print(f"  Empty query: {len(empty_results)} results")
    
    # Non-existent words
    missing_results = engine.keyword_search("xyzzyx foobar", k=5)
    print(f"  Non-existent words: {len(missing_results)} results")
    
    # Very short words (should be filtered)
    short_results = engine.keyword_search("a an the", k=5)
    print(f"  Short words: {len(short_results)} results")
    
    print("\n🎉 Keyword search tests complete!")

if __name__ == "__main__":
    test_keyword_search() 