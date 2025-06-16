#!/usr/bin/env python3
"""Test query preprocessing and optimization."""

import numpy as np
from core.search_engine import SearchEngine, SearchResult

def test_query_preprocessing():
    """Test query preprocessing and optimization features."""
    print("🔍 Testing Query Preprocessing...")
    
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
    
    # Test query preprocessing
    print("\n🔧 Testing query preprocessing:")
    
    test_queries = [
        "machine learning algorithms",  # Technical query
        "What is deep learning?",       # Question
        "error",                        # Single term
        "How do neural networks work and what are the main components?",  # Long query
        "bug fix solution",             # Exact match terms
        "ml",                          # Abbreviation
        "",                            # Empty query
        "  MACHINE LEARNING  "         # Needs normalization
    ]
    
    for query in test_queries:
        analysis = engine.preprocess_query(query)
        
        print(f"\n  Query: '{query}'")
        print(f"    Processed: '{analysis['processed']}'")
        print(f"    Tokens: {analysis['tokens']}")
        print(f"    Intent: {analysis['intent']}")
        print(f"    Suggested search: {analysis['search_type']}")
        if analysis['expansions']:
            print(f"    Expansions: {analysis['expansions']}")
    
    # Test intent detection
    print("\n🎯 Testing intent detection:")
    
    intent_tests = [
        ("What is machine learning?", "question"),
        ("neural network algorithm", "technical"),
        ("bug fix error", "exact_match"),
        ("learning", "single_term"),
        ("How do deep neural networks work with multiple layers for pattern recognition?", "long_query"),
        ("general search query", "general")
    ]
    
    for query, expected_intent in intent_tests:
        analysis = engine.preprocess_query(query)
        actual_intent = analysis['intent']
        status = "✅" if actual_intent == expected_intent else "❌"
        print(f"  {status} '{query}' → {actual_intent} (expected: {expected_intent})")
    
    # Test query expansion
    print("\n📈 Testing query expansion:")
    
    expansion_tests = [
        "ml algorithms",
        "ai and dl",
        "nlp processing",
        "cv techniques",
        "rl agents"
    ]
    
    for query in expansion_tests:
        analysis = engine.preprocess_query(query)
        if analysis['expansions']:
            print(f"  '{query}' → {analysis['expansions']}")
        else:
            print(f"  '{query}' → No expansions")
    
    # Test smart search
    print("\n🧠 Testing smart search:")
    
    smart_queries = [
        "machine learning",
        "What is neural network?",
        "ml",
        "error fix"
    ]
    
    for query in smart_queries:
        result = engine.smart_search(query, k=2)
        
        print(f"\n  Query: '{query}'")
        print(f"    Analysis: {result['analysis']['intent']} → {result['search_method']}")
        print(f"    Processing time: {result['processing_time']:.3f}s")
        print(f"    Results: {result['result_count']}")
        
        if result['results']:
            for i, res in enumerate(result['results'], 1):
                print(f"      {i}. [Score: {res.score:.3f}] {res.text[:40]}...")
    
    print("\n🎉 Query preprocessing tests complete!")

if __name__ == "__main__":
    test_query_preprocessing() 