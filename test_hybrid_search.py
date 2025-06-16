#!/usr/bin/env python3
"""Test hybrid search implementation."""

import numpy as np
from core.search_engine import SearchEngine, SearchResult

def test_hybrid_search():
    """Test hybrid search fusion algorithm."""
    print("🔍 Testing Hybrid Search...")
    
    # Test documents with varied content
    sample_docs = [
        {"id": "1", "text": "Machine learning algorithms learn patterns from data to make predictions"},
        {"id": "2", "text": "Natural language processing helps computers understand human language"},
        {"id": "3", "text": "Deep neural networks have multiple layers for complex pattern recognition"},
        {"id": "4", "text": "Computer vision enables machines to interpret and analyze visual information"},
        {"id": "5", "text": "Reinforcement learning agents learn through trial and error with rewards"}
    ]
    
    print(f"📝 Test corpus: {len(sample_docs)} documents")
    
    # Create embeddings (we need them for index building)
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
    
    # Test hybrid search
    print("\n🔍 Testing hybrid search:")
    
    test_queries = [
        "machine learning algorithms",
        "language processing",
        "neural networks",
        "computer vision",
        "reinforcement learning"
    ]
    
    for query in test_queries:
        results = engine.hybrid_search(query, k=3)
        
        print(f"\n  Query: '{query}'")
        if results:
            for i, result in enumerate(results, 1):
                print(f"    {i}. [Score: {result.score:.3f}] [{result.source}] {result.text[:50]}...")
        else:
            print("    No results found")
    
    # Test search method comparison
    print("\n📊 Comparing search methods:")
    
    comparison_query = "machine learning patterns"
    comparison = engine.compare_search_methods(comparison_query, k=3)
    
    print(f"\n  Query: '{comparison_query}'")
    
    for method, results in comparison.items():
        print(f"\n  {method.upper()} Results:")
        if results:
            for i, result in enumerate(results, 1):
                print(f"    {i}. [Score: {result.score:.3f}] {result.text[:50]}...")
        else:
            print("    No results found")
    
    # Test different weight configurations
    print("\n⚖️ Testing different weight configurations:")
    
    weight_configs = [
        (0.8, 0.2, "Semantic-heavy"),
        (0.5, 0.5, "Balanced"),
        (0.2, 0.8, "Keyword-heavy")
    ]
    
    test_query = "neural network layers"
    
    for sem_weight, key_weight, description in weight_configs:
        results = engine.hybrid_search(test_query, k=2, 
                                     semantic_weight=sem_weight,
                                     keyword_weight=key_weight)
        
        print(f"\n  {description} ({sem_weight:.1f}/{key_weight:.1f}):")
        if results:
            for i, result in enumerate(results, 1):
                print(f"    {i}. [Score: {result.score:.3f}] [{result.source}] {result.text[:40]}...")
    
    # Test unified search interface
    print("\n🎯 Testing unified search interface:")
    
    interface_query = "computer vision"
    
    for search_type in ['semantic', 'keyword', 'hybrid']:
        results = engine.search(interface_query, limit=2, search_type=search_type)
        
        print(f"\n  {search_type.upper()} via unified interface:")
        if results:
            for i, result in enumerate(results, 1):
                print(f"    {i}. [Score: {result.score:.3f}] [{result.source}] {result.text[:40]}...")
    
    # Test edge cases
    print("\n🧪 Testing edge cases:")
    
    # Empty query
    empty_results = engine.hybrid_search("", k=5)
    print(f"  Empty query: {len(empty_results)} results")
    
    # Invalid weights
    try:
        engine.hybrid_search("test", semantic_weight=0.7, keyword_weight=0.4)
        print("  Invalid weights: ERROR - should have failed!")
    except ValueError as e:
        print(f"  Invalid weights: ✅ Correctly caught error")
    
    print("\n🎉 Hybrid search tests complete!")

if __name__ == "__main__":
    test_hybrid_search() 