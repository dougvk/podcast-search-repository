#!/usr/bin/env python3
"""Test FAISS index setup and configuration."""

from core.search_engine import SearchEngine, SearchResult
import numpy as np
import pytest

def test_faiss_setup():
    """Test FAISS index configuration and building."""
    print("🔍 Testing FAISS Index Setup...")
    
    # Test full search engine
    print("\n🚀 Testing SearchEngine initialization:")
    
    engine = SearchEngine()
    print(f"  ✅ Engine initialized")
    print(f"  📊 Stats: {engine.get_stats()}")
    
    # Test index building with sample data
    print("\n🔨 Testing index building:")
    
    # Create sample embeddings and documents
    sample_docs = [
        {"id": "1", "text": "Machine learning is revolutionizing technology"},
        {"id": "2", "text": "Natural language processing enables AI communication"},
        {"id": "3", "text": "Deep learning models improve with more data"},
        {"id": "4", "text": "Computer vision helps machines understand images"},
        {"id": "5", "text": "Reinforcement learning teaches agents through rewards"}
    ]
    
    # Generate fake embeddings (384D normalized)
    n_docs = len(sample_docs)
    dimension = 384
    embeddings = np.random.randn(n_docs, dimension).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings.tolist()
    
    print(f"  📝 Generated {len(embeddings)} embeddings")
    print(f"  🎯 Dimension: {len(embeddings[0])}D")
    
    # Build index
    try:
        engine.build_index(embeddings, sample_docs)
        print(f"  ✅ Index built successfully")
        print(f"  📊 Final stats: {engine.get_stats()}")
        
        # Test search functionality
        print("\n🔍 Testing search functionality:")
        
        # Test semantic search
        results = engine.semantic_search("machine learning technology", k=3)
        print(f"  ✅ Semantic search returned {len(results)} results")
        
        # Test keyword search
        results = engine.keyword_search("learning", k=3)
        print(f"  ✅ Keyword search returned {len(results)} results")
        
        # Test hybrid search
        results = engine.hybrid_search("artificial intelligence", k=3)
        print(f"  ✅ Hybrid search returned {len(results)} results")
        
    except Exception as e:
        print(f"  ❌ Error during testing: {e}")
        pytest.fail(f"FAISS setup test failed: {e}")
    
    print("\n🎉 FAISS setup tests complete!")

if __name__ == "__main__":
    test_faiss_setup() 