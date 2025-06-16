#!/usr/bin/env python3
"""Test semantic search implementation."""

import numpy as np
from core.search_engine import SearchEngine, SearchResult
from core.embeddings import EmbeddingGenerator

def test_semantic_search():
    """Test semantic search with real embeddings."""
    print("🔍 Testing Semantic Search...")
    
    # Test documents about AI/ML topics
    sample_docs = [
        {"id": "1", "text": "Machine learning algorithms learn patterns from data to make predictions"},
        {"id": "2", "text": "Natural language processing helps computers understand human language"},
        {"id": "3", "text": "Deep neural networks have multiple layers for complex pattern recognition"},
        {"id": "4", "text": "Computer vision enables machines to interpret and analyze visual information"},
        {"id": "5", "text": "Reinforcement learning agents learn through trial and error with rewards"}
    ]
    
    print(f"📝 Test corpus: {len(sample_docs)} AI/ML documents")
    
    # Generate embeddings
    print("\n🧠 Generating embeddings...")
    with EmbeddingGenerator() as embedder:
        texts = [doc['text'] for doc in sample_docs]
        embeddings = embedder.generate_embeddings(texts)
        
        print(f"  ✅ Generated {len(embeddings)} embeddings ({len(embeddings[0])}D)")
        
        # Build search engine
        engine = SearchEngine()
        engine.build_index(embeddings, sample_docs)
        
        print(f"  ✅ Index built: {engine.get_stats()}")
        
        # Test semantic searches
        test_queries = [
            "deep learning neural networks",
            "language understanding NLP",
            "computer vision images",
            "reinforcement learning rewards",
            "machine learning patterns"
        ]
        
        print("\n🔍 Testing semantic searches:")
        for query in test_queries:
            results = engine.semantic_search(query, k=3, embedder=embedder)
            
            print(f"\n  Query: '{query}'")
            if results:
                for i, result in enumerate(results, 1):
                    print(f"    {i}. [Score: {result.score:.3f}] {result.text[:60]}...")
            else:
                print("    No results found")
        
        # Test edge cases
        print("\n🧪 Testing edge cases:")
        
        # Empty query
        empty_results = engine.semantic_search("", k=5, embedder=embedder)
        print(f"  Empty query: {len(empty_results)} results")
        
        # Single word query
        single_results = engine.semantic_search("learning", k=3, embedder=embedder)
        print(f"  Single word query: {len(single_results)} results")
        
        # Query without embedder (should create internal one)
        auto_results = engine.semantic_search("artificial intelligence", k=2)
        print(f"  Auto-embedder query: {len(auto_results)} results")
        
        print("\n🎉 Semantic search tests complete!")

if __name__ == "__main__":
    test_semantic_search() 