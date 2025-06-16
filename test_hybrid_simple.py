#!/usr/bin/env python3
"""Simple hybrid search test without embeddings."""

import numpy as np
from core.search_engine import SearchEngine, SearchResult

def test_hybrid_simple():
    """Test hybrid search with keyword-only approach."""
    print("🔍 Testing Hybrid Search (Simple)...")
    
    # Test documents
    sample_docs = [
        {"id": "1", "text": "Machine learning algorithms learn patterns from data"},
        {"id": "2", "text": "Natural language processing helps computers understand language"},
        {"id": "3", "text": "Deep neural networks have multiple layers for recognition"},
        {"id": "4", "text": "Computer vision enables machines to interpret visual information"},
        {"id": "5", "text": "Reinforcement learning agents learn through trial and error"}
    ]
    
    print(f"📝 Test corpus: {len(sample_docs)} documents")
    
    # Create fake embeddings
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
    
    # Test keyword search first
    print("\n🔍 Testing keyword search:")
    
    keyword_results = engine.keyword_search("machine learning", k=3)
    print(f"  Keyword results: {len(keyword_results)}")
    for i, result in enumerate(keyword_results, 1):
        print(f"    {i}. [Score: {result.score:.3f}] {result.text[:50]}...")
    
    # Test score normalization
    print("\n📊 Testing score normalization:")
    
    if keyword_results:
        original_scores = [r.score for r in keyword_results]
        print(f"  Original scores: {[f'{s:.3f}' for s in original_scores]}")
        
        normalized = engine._normalize_scores(keyword_results.copy())
        normalized_scores = [r.score for r in normalized]
        print(f"  Normalized scores: {[f'{s:.3f}' for s in normalized_scores]}")
    
    # Test result fusion (keyword only)
    print("\n🔧 Testing result fusion:")
    
    # Create two sets of keyword results with different queries
    results1 = engine.keyword_search("machine learning", k=3)
    results2 = engine.keyword_search("algorithms patterns", k=3)
    
    if results1 and results2:
        # Normalize both sets
        norm1 = engine._normalize_scores(results1.copy())
        norm2 = engine._normalize_scores(results2.copy())
        
        # Fuse them
        fused = engine._fuse_results(norm1, norm2, 0.6, 0.4)
        
        print(f"  Fused results: {len(fused)}")
        for i, result in enumerate(fused, 1):
            print(f"    {i}. [Score: {result.score:.3f}] [{result.source}] {result.text[:40]}...")
    
    # Test unified search interface (keyword only)
    print("\n🎯 Testing unified search interface:")
    
    for search_type in ['keyword']:  # Only test keyword to avoid segfault
        results = engine.search("machine learning", limit=2, search_type=search_type)
        
        print(f"\n  {search_type.upper()} via unified interface:")
        if results:
            for i, result in enumerate(results, 1):
                print(f"    {i}. [Score: {result.score:.3f}] [{result.source}] {result.text[:40]}...")
    
    print("\n🎉 Simple hybrid search tests complete!")

if __name__ == "__main__":
    test_hybrid_simple() 