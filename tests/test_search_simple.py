#!/usr/bin/env python3
"""Simple test for FAISS index without embeddings."""

import sys
import os
import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.search_engine import SearchEngine

def test_faiss_simple():
    """Test FAISS index with fake embeddings."""
    print("🔍 Testing FAISS Index (Simple)...")
    
    # Test search engine initialization
    print("\n🚀 Testing SearchEngine:")
    engine = SearchEngine()
    print(f"  ✅ Engine initialized")
    
    # Create fake embeddings (384D)
    print("\n🔨 Testing with fake embeddings:")
    n_docs = 5
    dimension = 384
    
    # Generate random normalized embeddings
    embeddings = np.random.randn(n_docs, dimension).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # Normalize
    embeddings = embeddings.tolist()  # Convert to list
    
    documents = [
        {"id": f"{i}", "text": f"Document {i} with sample content"}
        for i in range(n_docs)
    ]
    
    print(f"  📝 Created {len(embeddings)} fake embeddings")
    print(f"  🎯 Dimension: {len(embeddings[0])}D")
    
    # Build index
    try:
        engine.build_index(embeddings, documents)
        print(f"  ✅ Index built successfully")
        print(f"  📊 Stats: {engine.get_stats()}")
        
        # Test basic search functionality
        print("\n🔍 Testing search:")
        
        # Test with a simple query 
        results = engine.search("sample content", limit=3, search_type='semantic')
        print(f"  ✅ Search returned {len(results)} results")
        
        if results:
            print(f"  📊 Top result score: {results[0].score:.3f}")
        
    except Exception as e:
        print(f"  ❌ Error building index: {e}")
        pytest.fail(f"Simple FAISS test failed: {e}")
    
    print("\n🎉 Simple FAISS test complete!")

if __name__ == "__main__":
    test_faiss_simple() 