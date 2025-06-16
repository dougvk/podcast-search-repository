#!/usr/bin/env python3
"""Test FAISS index setup and configuration."""

from core.search_engine import SearchEngine, FAISSConfig, SearchResult
from core.embeddings import EmbeddingGenerator
import numpy as np

def test_faiss_setup():
    """Test FAISS index configuration and building."""
    print("🔍 Testing FAISS Index Setup...")
    
    # Test index type selection
    print("\n📊 Testing index type selection:")
    
    # Small corpus - should get FlatIP
    small_index = FAISSConfig.get_optimal_index(50, 384)
    print(f"  Small corpus (50): {type(small_index).__name__}")
    
    # Medium corpus - should get IVFFlat  
    medium_index = FAISSConfig.get_optimal_index(500, 384)
    print(f"  Medium corpus (500): {type(medium_index).__name__}")
    
    # Large corpus - should get HNSW
    large_index = FAISSConfig.get_optimal_index(1500, 384)
    print(f"  Large corpus (1500): {type(large_index).__name__}")
    
    # Test full search engine
    print("\n🚀 Testing SearchEngine initialization:")
    
    engine = SearchEngine()
    print(f"  ✅ Engine initialized: {engine.is_ready}")
    print(f"  📊 Stats: {engine.get_stats()}")
    
    # Test index building with sample data
    print("\n🔨 Testing index building:")
    
    # Generate sample embeddings and documents
    with EmbeddingGenerator() as embedder:
        sample_docs = [
            {"id": "1", "text": "Machine learning is revolutionizing technology"},
            {"id": "2", "text": "Natural language processing enables AI communication"},
            {"id": "3", "text": "Deep learning models improve with more data"},
            {"id": "4", "text": "Computer vision helps machines understand images"},
            {"id": "5", "text": "Reinforcement learning teaches agents through rewards"}
        ]
        
        # Generate embeddings
        texts = [doc['text'] for doc in sample_docs]
        embeddings = embedder.generate_embeddings(texts)
        
        print(f"  📝 Generated {len(embeddings)} embeddings")
        print(f"  🎯 Dimension: {len(embeddings[0])}D")
        
        # Build index
        engine.build_index(embeddings, sample_docs)
        
        print(f"  ✅ Index built successfully")
        print(f"  📊 Final stats: {engine.get_stats()}")
        
        # Test persistence
        print("\n💾 Testing index persistence:")
        
        # Create new engine to test loading
        engine2 = SearchEngine()
        print(f"  ✅ Loaded from disk: {engine2.is_ready}")
        print(f"  📊 Loaded stats: {engine2.get_stats()}")
        
        print("\n🎉 FAISS setup tests complete!")

if __name__ == "__main__":
    test_faiss_setup() 