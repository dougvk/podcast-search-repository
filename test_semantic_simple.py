#!/usr/bin/env python3
"""Simple semantic search test with fake embeddings."""

import numpy as np
from core.search_engine import SearchEngine, SearchResult

def test_semantic_search_simple():
    """Test semantic search with fake embeddings."""
    print("🔍 Testing Semantic Search (Simple)...")
    
    # Test documents
    sample_docs = [
        {"id": "1", "text": "Machine learning algorithms learn patterns from data"},
        {"id": "2", "text": "Natural language processing helps computers understand language"},
        {"id": "3", "text": "Deep neural networks have multiple layers for recognition"},
        {"id": "4", "text": "Computer vision enables machines to interpret visual information"},
        {"id": "5", "text": "Reinforcement learning agents learn through trial and error"}
    ]
    
    print(f"📝 Test corpus: {len(sample_docs)} documents")
    
    # Create fake embeddings (normalized)
    np.random.seed(42)  # Reproducible results
    n_docs = len(sample_docs)
    dimension = 384
    
    # Generate clustered embeddings to simulate semantic similarity
    embeddings = []
    
    # Create topic clusters
    topics = [
        np.array([1.0, 0.5, 0.8, 0.2, 0.1]),  # ML patterns topic
        np.array([0.3, 1.0, 0.4, 0.7, 0.2]),  # Language topic  
        np.array([0.8, 0.3, 1.0, 0.1, 0.6]),  # Neural networks topic
        np.array([0.2, 0.8, 0.3, 1.0, 0.4]),  # Vision topic
        np.array([0.4, 0.2, 0.6, 0.3, 1.0])   # RL topic
    ]
    
    for i, topic_vec in enumerate(topics):
        # Create 384D embedding with topic signature + noise
        embedding = np.random.randn(dimension).astype(np.float32) * 0.1
        # Embed topic signature in first 5 dimensions
        embedding[:5] = topic_vec
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding.tolist())
    
    print(f"  ✅ Created {len(embeddings)} fake embeddings ({len(embeddings[0])}D)")
    
    # Build search index
    engine = SearchEngine()
    engine.build_index(embeddings, sample_docs)
    
    print(f"  ✅ Index built: {engine.get_stats()}")
    
    # Test semantic search with fake query embeddings
    print("\n🔍 Testing semantic searches:")
    
    # Create query embeddings similar to different topics
    test_queries = [
        ("machine learning patterns", [1.0, 0.4, 0.7, 0.1, 0.0]),  # Similar to doc 1
        ("language understanding", [0.2, 1.0, 0.3, 0.8, 0.1]),    # Similar to doc 2
        ("neural network layers", [0.7, 0.2, 1.0, 0.0, 0.5]),    # Similar to doc 3
        ("computer vision images", [0.1, 0.7, 0.2, 1.0, 0.3]),   # Similar to doc 4
        ("reinforcement rewards", [0.3, 0.1, 0.5, 0.2, 1.0])     # Similar to doc 5
    ]
    
    for query_text, topic_signature in test_queries:
        # Create fake query embedding
        query_embedding = np.random.randn(dimension).astype(np.float32) * 0.1
        query_embedding[:5] = np.array(topic_signature)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Use direct similarity search  
        try:
            results = engine._similarity_search_with_embeddings([query_embedding.tolist()], k=3)
            
            print(f"\n  Query: '{query_text}'")
            if results:
                for i, (score, doc_idx) in enumerate(results[:3], 1):
                    if doc_idx < len(sample_docs):
                        doc = sample_docs[doc_idx]
                        print(f"    {i}. [Score: {score:.3f}] {doc['text'][:50]}...")
            else:
                print("    No results found")
                
        except Exception as e:
            print(f"    Error: {e}")
    
    print("\n🎉 Simple semantic search tests complete!")

if __name__ == "__main__":
    test_semantic_search_simple() 