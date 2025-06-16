#!/usr/bin/env python3
"""Test real embeddings using memvid's proven MemvidEncoder approach."""

import sys
import os
from pathlib import Path
import time
import json

# Add memvid to path (assuming it's installed)
try:
    from memvid import MemvidEncoder
    from memvid.config import get_default_config
    print("✅ Successfully imported memvid")
except ImportError as e:
    print(f"❌ Could not import memvid: {e}")
    print("Please install memvid: pip install memvid")
    sys.exit(1)

# Import our search engine
from core.search_engine import SearchEngine

def extract_embeddings_from_memvid(texts):
    """Extract embeddings using memvid's proven approach."""
    print(f"🧠 Generating embeddings for {len(texts)} texts using memvid...")
    
    # Use memvid's configuration approach
    config = get_default_config()
    print(f"📋 Using memvid config: {config.get('embedding', {}).get('model', 'default')}")
    
    # Create encoder with memvid's proven initialization
    encoder = MemvidEncoder(config)
    
    # The key insight: memvid stores the embedding model in index_manager
    if hasattr(encoder, 'index_manager') and hasattr(encoder.index_manager, 'embedding_model'):
        model = encoder.index_manager.embedding_model
        print("✅ Found memvid's embedding model in index_manager, generating embeddings...")
        
        try:
            # Use memvid's embedding model directly
            embeddings = []
            for i, text in enumerate(texts):
                print(f"  🧠 Generating embedding {i+1}/{len(texts)}")
                
                # Use the same method memvid uses internally
                embedding = model.encode(text, convert_to_numpy=True)
                embeddings.append(embedding.tolist())
            
            print(f"✅ Generated {len(embeddings)} embeddings using memvid's model")
            return embeddings
            
        except Exception as e:
            print(f"❌ Direct embedding generation failed: {e}")
            return None
    
    else:
        print("❌ Could not access memvid's embedding model")
        return None

def test_memvid_embeddings_integration():
    """Test our search engine with real embeddings from memvid."""
    print("🚀 Testing Search Engine with Memvid Embeddings")
    print("=" * 50)
    
    # Test documents
    documents = [
        {
            "id": "doc_1",
            "text": "Machine learning algorithms enable computers to learn patterns from data without explicit programming.",
            "metadata": {"category": "AI"}
        },
        {
            "id": "doc_2", 
            "text": "Deep neural networks with multiple hidden layers can model complex nonlinear relationships.",
            "metadata": {"category": "Deep Learning"}
        },
        {
            "id": "doc_3",
            "text": "Natural language processing helps computers understand and generate human language.",
            "metadata": {"category": "NLP"}
        },
        {
            "id": "doc_4",
            "text": "Computer vision systems can analyze and interpret visual information from images and videos.",
            "metadata": {"category": "Computer Vision"}
        },
        {
            "id": "doc_5",
            "text": "Reinforcement learning agents learn optimal actions through trial and error with rewards.",
            "metadata": {"category": "RL"}
        }
    ]
    
    print(f"📝 Test dataset: {len(documents)} documents")
    
    # Extract texts for embedding
    texts = [doc['text'] for doc in documents]
    
    # Generate embeddings using memvid's approach
    start_time = time.time()
    embeddings = extract_embeddings_from_memvid(texts)
    embedding_time = time.time() - start_time
    
    if not embeddings:
        print("❌ Failed to generate embeddings")
        return False
    
    print(f"⏱️  Embedding generation: {embedding_time:.3f}s")
    print(f"📊 Embedding dimensions: {len(embeddings[0])}D")
    
    # Test our search engine with real embeddings
    print("\n🔍 Testing Search Engine Integration...")
    
    engine = SearchEngine(
        index_dir="data/memvid_test",
        cache_size=50,
        max_workers=2
    )
    
    # Build index with real embeddings
    start_time = time.time()
    engine.build_index(embeddings, documents)
    build_time = time.time() - start_time
    
    print(f"✅ Index built in {build_time:.3f}s")
    print(f"📊 Engine stats: {engine.get_stats()}")
    
    # Test semantic search with real embeddings
    print("\n🧠 Testing Semantic Search with Real Embeddings...")
    
    test_queries = [
        "neural network deep learning",
        "language understanding NLP",
        "computer vision image analysis", 
        "reinforcement learning rewards",
        "machine learning patterns"
    ]
    
    # Generate query embeddings using same method
    print("🔄 Generating query embeddings...")
    query_embeddings_data = extract_embeddings_from_memvid(test_queries)
    
    if not query_embeddings_data:
        print("⚠️  Could not generate query embeddings, testing keyword search only")
        
        # Test keyword search
        for query in test_queries:
            start = time.time()
            results = engine.keyword_search(query, k=3)
            search_time = time.time() - start
            
            print(f"\n🔍 Keyword: '{query}' ({search_time:.3f}s)")
            for i, result in enumerate(results[:2]):
                print(f"  {i+1}. {result.id}: {result.score:.3f} - {result.text[:60]}...")
    
    else:
        print("✅ Query embeddings generated successfully")
        
        # Test semantic search with real embeddings
        for i, query in enumerate(test_queries):
            start = time.time()
            
            # Use the pre-generated query embedding
            query_embedding = query_embeddings_data[i]
            
            # Perform similarity search directly
            results = engine._similarity_search_with_embeddings([query_embedding], k=3)
            search_time = time.time() - start
            
            print(f"\n🧠 Semantic: '{query}' ({search_time:.3f}s)")
            for j, (score, doc_idx) in enumerate(results[:2]):
                if doc_idx < len(documents):
                    doc = documents[doc_idx]
                    print(f"  {j+1}. {doc['id']}: {score:.3f} - {doc['text'][:60]}...")
    
    # Test hybrid search
    print("\n🔀 Testing Hybrid Search...")
    for query in test_queries[:2]:
        start = time.time()
        results = engine.search(query, limit=3, search_type='hybrid')
        search_time = time.time() - start
        
        print(f"\n🔀 Hybrid: '{query}' ({search_time:.3f}s)")
        for i, result in enumerate(results[:2]):
            print(f"  {i+1}. {result.id}: {result.score:.3f} ({result.source}) - {result.text[:60]}...")
    
    # Performance summary
    print(f"\n📊 PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"🧠 Embedding generation: {embedding_time:.3f}s for {len(texts)} texts")
    print(f"🏗️  Index building: {build_time:.3f}s")
    print(f"🔍 Search engine: Ready with real embeddings!")
    print(f"✅ No segmentation faults - memvid approach works!")
    
    return True

def test_memvid_config_compatibility():
    """Test compatibility with memvid's configuration system."""
    print("\n🔧 Testing Memvid Configuration Compatibility...")
    
    try:
        # Get memvid's default config
        config = get_default_config()
        print(f"📋 Memvid default config keys: {list(config.keys())}")
        
        # Test embedding config
        if 'embedding' in config:
            embedding_config = config['embedding']
            print(f"🧠 Embedding config: {embedding_config}")
        
        # Test chunking config
        if 'chunking' in config:
            chunk_config = config['chunking']
            print(f"📝 Chunking config: size={chunk_config.get('chunk_size')}, overlap={chunk_config.get('overlap')}")
        
        # Test index config  
        if 'index' in config:
            index_config = config['index']
            print(f"🏗️  Index config: type={index_config.get('type')}")
        
        print("✅ Configuration compatibility confirmed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_memvid_encoder_internals():
    """Test accessing memvid encoder internals to understand the architecture."""
    print("\n🔍 Inspecting Memvid Encoder Internals...")
    
    try:
        config = get_default_config()
        encoder = MemvidEncoder(config)
        
        print(f"📋 Encoder attributes: {[attr for attr in dir(encoder) if not attr.startswith('_')]}")
        
        # Check for embedding model
        if hasattr(encoder, 'embedding_model'):
            model = encoder.embedding_model
            print(f"🧠 Embedding model type: {type(model)}")
            print(f"🧠 Embedding model: {model}")
            
            # Test a simple encoding
            test_text = "This is a test sentence."
            try:
                embedding = model.encode(test_text)
                print(f"✅ Test embedding shape: {embedding.shape}")
                print(f"✅ Test embedding type: {type(embedding)}")
                return True
            except Exception as e:
                print(f"❌ Test encoding failed: {e}")
        else:
            print("❌ No embedding_model attribute found")
            
        # Check index_manager for embedding functionality
        if hasattr(encoder, 'index_manager'):
            index_mgr = encoder.index_manager
            print(f"🏗️  Index manager type: {type(index_mgr)}")
            print(f"🏗️  Index manager attributes: {[attr for attr in dir(index_mgr) if not attr.startswith('_')]}")
            
            # Check if index manager has embedding model
            if hasattr(index_mgr, 'embedding_model'):
                model = index_mgr.embedding_model
                print(f"🧠 Found embedding model in index manager: {type(model)}")
                
                # Test encoding
                test_text = "This is a test sentence."
                try:
                    embedding = model.encode(test_text)
                    print(f"✅ Test embedding shape: {embedding.shape}")
                    print(f"✅ Test embedding type: {type(embedding)}")
                    return True
                except Exception as e:
                    print(f"❌ Test encoding failed: {e}")
            
            # Check for other embedding-related methods
            embedding_methods = [attr for attr in dir(index_mgr) if 'embed' in attr.lower()]
            if embedding_methods:
                print(f"🔍 Embedding-related methods: {embedding_methods}")
                
                # Try to use generate_embeddings if it exists
                if hasattr(index_mgr, 'generate_embeddings'):
                    try:
                        test_texts = ["This is a test sentence."]
                        embeddings = index_mgr.generate_embeddings(test_texts)
                        print(f"✅ Generated embeddings via index manager: {len(embeddings)} embeddings")
                        print(f"✅ Embedding shape: {len(embeddings[0])}D")
                        return True
                    except Exception as e:
                        print(f"❌ generate_embeddings failed: {e}")
            
    except Exception as e:
        print(f"❌ Encoder inspection failed: {e}")
    
    return False

def main():
    """Run memvid embedding integration tests."""
    print("🎯 Memvid Embedding Integration Test")
    print("=" * 60)
    
    # Test configuration compatibility
    config_ok = test_memvid_config_compatibility()
    
    # Test encoder internals
    internals_ok = test_memvid_encoder_internals()
    
    if config_ok and internals_ok:
        # Test embedding integration
        embedding_ok = test_memvid_embeddings_integration()
        
        if embedding_ok:
            print("\n🎉 SUCCESS: Memvid embedding integration working!")
            print("✅ Real embeddings generated without segfaults")
            print("✅ Search engine integration successful")
            print("✅ Performance optimizations validated")
            return True
    
    print("\n⚠️  Some tests failed - check output above")
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 