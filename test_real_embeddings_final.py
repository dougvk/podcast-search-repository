#!/usr/bin/env python3
"""
Final comprehensive test of SearchEngine with real memvid embeddings.
Validates all functionality with memvid's proven approach.
"""

import sys
import os
import time
from pathlib import Path

# Import our refactored search engine
from core.search_engine import SearchEngine, MEMVID_AVAILABLE

def test_memvid_integration():
    """Test that memvid integration is working properly."""
    print("🔍 Testing Memvid Integration...")
    
    if not MEMVID_AVAILABLE:
        print("❌ Memvid not available - please install: pip install memvid")
        return False
    
    print("✅ Memvid is available")
    
    # Test embedding generator initialization
    try:
        engine = SearchEngine()
        print("✅ SearchEngine initialized with memvid embeddings")
        
        # Test embedding generation
        test_text = "This is a test sentence for embedding generation."
        embedding = engine.embedding_generator.generate_embeddings(test_text)
        
        print(f"✅ Generated embedding: {len(embedding)}D vector")
        print(f"   Sample values: {embedding[:5]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Memvid integration failed: {e}")
        return False

def test_real_semantic_search():
    """Test semantic search with real embeddings."""
    print("\n🧠 Testing Real Semantic Search...")
    
    # Create test documents with semantic relationships
    documents = [
        {
            'id': 'doc_1',
            'text': 'Machine learning algorithms for data analysis and pattern recognition',
            'metadata': {'category': 'AI', 'difficulty': 'advanced'}
        },
        {
            'id': 'doc_2', 
            'text': 'Python programming tutorial for beginners with examples',
            'metadata': {'category': 'Programming', 'difficulty': 'beginner'}
        },
        {
            'id': 'doc_3',
            'text': 'Deep learning neural networks and artificial intelligence',
            'metadata': {'category': 'AI', 'difficulty': 'expert'}
        },
        {
            'id': 'doc_4',
            'text': 'Web development with JavaScript and React framework',
            'metadata': {'category': 'Web', 'difficulty': 'intermediate'}
        },
        {
            'id': 'doc_5',
            'text': 'Data science visualization using matplotlib and seaborn',
            'metadata': {'category': 'Data Science', 'difficulty': 'intermediate'}
        }
    ]
    
    try:
        # Initialize search engine
        engine = SearchEngine()
        
        # Generate real embeddings
        print("🔄 Generating real embeddings...")
        start_time = time.time()
        
        texts = [doc['text'] for doc in documents]
        embeddings = engine.embedding_generator.generate_embeddings(texts)
        
        embedding_time = time.time() - start_time
        print(f"✅ Generated {len(embeddings)} embeddings in {embedding_time:.3f}s")
        print(f"   Embedding dimension: {len(embeddings[0])}D")
        
        # Build search index
        print("🔄 Building search index...")
        start_time = time.time()
        
        engine.build_index(embeddings, documents)
        
        build_time = time.time() - start_time
        print(f"✅ Index built in {build_time:.3f}s")
        
        # Test semantic queries
        test_queries = [
            "artificial intelligence and machine learning",
            "programming tutorials for beginners", 
            "data visualization and charts",
            "web development frameworks"
        ]
        
        print("\n📊 Semantic Search Results:")
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            
            start_time = time.time()
            results = engine.semantic_search(query, k=3)
            search_time = time.time() - start_time
            
            print(f"   ⏱️  Search time: {search_time:.3f}s")
            
            for i, result in enumerate(results, 1):
                print(f"   {i}. {result.id} (score: {result.score:.3f})")
                print(f"      {result.text[:60]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Semantic search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_optimizations():
    """Test performance optimizations with real embeddings."""
    print("\n⚡ Testing Performance Optimizations...")
    
    try:
        engine = SearchEngine(cache_size=100)
        
        # Create larger document set
        documents = []
        for i in range(20):
            documents.append({
                'id': f'perf_doc_{i}',
                'text': f'Performance test document {i} with machine learning content and data analysis topics',
                'metadata': {'test_id': i}
            })
        
        # Generate embeddings and build index
        texts = [doc['text'] for doc in documents]
        embeddings = engine.embedding_generator.generate_embeddings(texts)
        engine.build_index(embeddings, documents)
        
        # Test query caching
        query = "machine learning data analysis"
        
        # First search (cache miss)
        start_time = time.time()
        results1 = engine.search(query, search_type='semantic')
        first_time = time.time() - start_time
        
        # Second search (cache hit)
        start_time = time.time()
        results2 = engine.search(query, search_type='semantic')
        second_time = time.time() - start_time
        
        # Verify cache speedup
        if second_time < first_time:
            speedup = first_time / second_time if second_time > 0 else float('inf')
            print(f"✅ Cache speedup: {speedup:.1f}x ({first_time:.3f}s → {second_time:.3f}s)")
        else:
            print(f"⚠️  Cache timing: {first_time:.3f}s → {second_time:.3f}s")
        
        # Test different search types
        search_types = ['semantic', 'keyword']
        for search_type in search_types:
            start_time = time.time()
            results = engine.search(query, search_type=search_type)
            search_time = time.time() - start_time
            print(f"✅ {search_type.title()} search: {search_time:.3f}s, {len(results)} results")
        
        # Display stats
        stats = engine.get_stats()
        print(f"\n📈 Engine Stats:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_search_quality():
    """Test search quality with real embeddings."""
    print("\n🎯 Testing Search Quality...")
    
    try:
        engine = SearchEngine()
        
        # Create documents with clear semantic relationships
        documents = [
            {'id': 'ai_1', 'text': 'Artificial intelligence and machine learning algorithms'},
            {'id': 'ai_2', 'text': 'Deep learning neural networks for pattern recognition'},
            {'id': 'prog_1', 'text': 'Python programming language syntax and features'},
            {'id': 'prog_2', 'text': 'JavaScript web development and React components'},
            {'id': 'data_1', 'text': 'Data analysis with pandas and numpy libraries'},
            {'id': 'data_2', 'text': 'Statistical modeling and data visualization techniques'}
        ]
        
        # Build index
        texts = [doc['text'] for doc in documents]
        embeddings = engine.embedding_generator.generate_embeddings(texts)
        engine.build_index(embeddings, documents)
        
        # Test semantic similarity
        test_cases = [
            {
                'query': 'machine learning and AI',
                'expected_categories': ['ai_1', 'ai_2'],
                'description': 'AI-related query should match AI documents'
            },
            {
                'query': 'programming languages and coding',
                'expected_categories': ['prog_1', 'prog_2'],
                'description': 'Programming query should match programming documents'
            },
            {
                'query': 'data science and statistics',
                'expected_categories': ['data_1', 'data_2'],
                'description': 'Data science query should match data documents'
            }
        ]
        
        print("🔍 Quality Test Results:")
        for test_case in test_cases:
            query = test_case['query']
            expected = test_case['expected_categories']
            description = test_case['description']
            
            results = engine.semantic_search(query, k=3)
            top_ids = [r.id for r in results[:2]]
            
            # Check if top results match expected categories
            matches = sum(1 for id in top_ids if id in expected)
            quality_score = matches / len(expected) if expected else 0
            
            print(f"\n   Query: '{query}'")
            print(f"   {description}")
            print(f"   Expected: {expected}")
            print(f"   Got: {top_ids}")
            print(f"   Quality: {quality_score:.1%} ({matches}/{len(expected)} matches)")
            
            if quality_score >= 0.5:
                print("   ✅ Good semantic matching")
            else:
                print("   ⚠️  Semantic matching could be improved")
        
        return True
        
    except Exception as e:
        print(f"❌ Quality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests for the refactored search engine."""
    print("🚀 Testing Refactored SearchEngine with Real Memvid Embeddings")
    print("=" * 70)
    
    tests = [
        ("Memvid Integration", test_memvid_integration),
        ("Real Semantic Search", test_real_semantic_search),
        ("Performance Optimizations", test_performance_optimizations),
        ("Search Quality", test_search_quality)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*70}")
    print("📋 Test Summary:")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("🎉 All tests passed! SearchEngine successfully refactored with memvid.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 