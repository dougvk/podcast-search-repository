#!/usr/bin/env python3
"""Simple test for FAISS index without embeddings."""

import numpy as np
from core.search_engine import SearchEngine, FAISSConfig

def test_faiss_simple():
    """Test FAISS index with fake embeddings."""
    print("🔍 Testing FAISS Index (Simple)...")
    
    # Test index type selection
    print("\n📊 Testing index type selection:")
    index = FAISSConfig.get_optimal_index(50, 384)
    print(f"  Index type: {type(index).__name__}")
    
    # Test search engine initialization
    print("\n🚀 Testing SearchEngine:")
    engine = SearchEngine()
    print(f"  ✅ Engine initialized: {engine.is_ready}")
    
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
    except Exception as e:
        print(f"  ❌ Error building index: {e}")
    
    print("\n🎉 Simple FAISS test complete!")

if __name__ == "__main__":
    test_faiss_simple() 