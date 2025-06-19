
Directory structure:
└── README.md


Files Content:

================================================
FILE: README.md
================================================
# Memvid - Video-Based AI Memory 🧠📹

**The lightweight, game-changing solution for AI memory at scale**

[![PyPI version](https://badge.fury.io/py/memvid.svg)](https://pypi.org/project/memvid/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Memvid revolutionizes AI memory management by encoding text data into videos, enabling **lightning-fast semantic search** across millions of text chunks with **sub-second retrieval times**. Unlike traditional vector databases that consume massive amounts of RAM and storage, Memvid compresses your knowledge base into compact video files while maintaining instant access to any piece of information.

## 🎥 Demo

https://github.com/user-attachments/assets/ec550e93-e9c4-459f-a8a1-46e122b5851e



## ✨ Key Features

- 🎥 **Video-as-Database**: Store millions of text chunks in a single MP4 file
- 🔍 **Semantic Search**: Find relevant content using natural language queries
- 💬 **Built-in Chat**: Conversational interface with context-aware responses
- 📚 **PDF Support**: Direct import and indexing of PDF documents
- 🚀 **Fast Retrieval**: Sub-second search across massive datasets
- 💾 **Efficient Storage**: 10x compression compared to traditional databases
- 🔌 **Pluggable LLMs**: Works with OpenAI, Anthropic, or local models
- 🌐 **Offline-First**: No internet required after video generation
- 🔧 **Simple API**: Get started with just 3 lines of code

## 🎯 Use Cases

- **📖 Digital Libraries**: Index thousands of books in a single video file
- **🎓 Educational Content**: Create searchable video memories of course materials
- **📰 News Archives**: Compress years of articles into manageable video databases
- **💼 Corporate Knowledge**: Build company-wide searchable knowledge bases
- **🔬 Research Papers**: Quick semantic search across scientific literature
- **📝 Personal Notes**: Transform your notes into a searchable AI assistant

## 🚀 Why Memvid?

### Game-Changing Innovation
- **Video as Database**: Store millions of text chunks in a single MP4 file
- **Instant Retrieval**: Sub-second semantic search across massive datasets
- **10x Storage Efficiency**: Video compression reduces memory footprint dramatically
- **Zero Infrastructure**: No database servers, just files you can copy anywhere
- **Offline-First**: Works completely offline once videos are generated

### Lightweight Architecture
- **Minimal Dependencies**: Core functionality in ~1000 lines of Python
- **CPU-Friendly**: Runs efficiently without GPU requirements
- **Portable**: Single video file contains your entire knowledge base
- **Streamable**: Videos can be streamed from cloud storage

## 📦 Installation

### Quick Install
```bash
pip install memvid
```

### For PDF Support
```bash
pip install memvid PyPDF2
```

### Recommended Setup (Virtual Environment)
```bash
# Create a new project directory
mkdir my-memvid-project
cd my-memvid-project

# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install memvid
pip install memvid

# For PDF support:
pip install PyPDF2
```

## 🎯 Quick Start

### Basic Usage
```python
from memvid import MemvidEncoder, MemvidChat

# Create video memory from text chunks
chunks = ["Important fact 1", "Important fact 2", "Historical event details"]
encoder = MemvidEncoder()
encoder.add_chunks(chunks)
encoder.build_video("memory.mp4", "memory_index.json")

# Chat with your memory
chat = MemvidChat("memory.mp4", "memory_index.json")
chat.start_session()
response = chat.chat("What do you know about historical events?")
print(response)
```

### Building Memory from Documents
```python
from memvid import MemvidEncoder
import os

# Load documents with custom chunking
config = {
    "chunking": {
        "chunk_size": 512,
        "overlap": 50
    }
}
encoder = MemvidEncoder(config=config)

# Add text files
for file in os.listdir("documents"):
    with open(f"documents/{file}", "r") as f:
        encoder.add_text(f.read())

# Build optimized video
encoder.build_video(
    "knowledge_base.mp4",
    "knowledge_index.json",
    codec='h265'  # Use H.265 codec for better compression
)
```

### Advanced Search & Retrieval
```python
from memvid import MemvidRetriever

# Initialize retriever
retriever = MemvidRetriever("knowledge_base.mp4", "knowledge_index.json")

# Semantic search
results = retriever.search("machine learning algorithms", top_k=5)
for chunk, score in results:
    print(f"Score: {score:.3f} | {chunk[:100]}...")

# Get context window
context = retriever.get_context("explain neural networks", max_tokens=2000)
print(context)
```

### Interactive Chat Interface
```python
from memvid import MemvidChat

# Start interactive chat session
chat = MemvidChat("knowledge_base.mp4", "knowledge_index.json")
chat.start_session()
response = chat.chat("Your question here")
print(response)
```

### Testing with file_chat.py
The `examples/file_chat.py` script provides a comprehensive way to test Memvid with your own documents:

```bash
# Process a directory of documents
python examples/file_chat.py --input-dir /path/to/documents --provider google

# Process specific files
python examples/file_chat.py --files doc1.txt doc2.pdf --provider openai

# Use H.265 compression (requires Docker)
python examples/file_chat.py --input-dir docs/ --codec h265 --provider google

# Custom chunking for large documents
python examples/file_chat.py --files large.pdf --chunk-size 2048 --overlap 32 --provider google

# Load existing memory
python examples/file_chat.py --load-existing output/my_memory --provider google
```

### Complete Example: Chat with a PDF Book
```bash
# 1. Create a new directory and set up environment
mkdir book-chat-demo
cd book-chat-demo
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install memvid PyPDF2

# 3. Create book_chat.py
cat > book_chat.py << 'EOF'
from memvid import MemvidEncoder, chat_with_memory
import os

# Your PDF file
book_pdf = "book.pdf"  # Replace with your PDF path

# Build video memory
encoder = MemvidEncoder()
encoder.add_pdf(book_pdf)
encoder.build_video("book_memory.mp4", "book_index.json")

# Chat with the book
api_key = os.getenv("OPENAI_API_KEY")  # Optional: for AI responses
chat_with_memory("book_memory.mp4", "book_index.json", api_key=api_key)
EOF

# 4. Run it
export OPENAI_API_KEY="your-api-key"  # Optional
python book_chat.py
```

## 🛠️ Advanced Configuration

### Custom Embeddings
```python
# Use custom embedding model via config
config = {
    "embedding": {
        "model_name": "sentence-transformers/all-mpnet-base-v2"
    }
}
encoder = MemvidEncoder(config=config)
```

### Video Optimization
```python
# For maximum compression
encoder.build_video(
    "compressed.mp4",
    "index.json",
    codec='h265'  # H.265 codec for better compression
)
```

### Distributed Processing
```python
# Process large datasets
encoder = MemvidEncoder()
encoder.add_chunks(massive_chunk_list)
```

## 🐛 Troubleshooting

### Common Issues

**ModuleNotFoundError: No module named 'memvid'**
```bash
# Make sure you're using the right Python
which python  # Should show your virtual environment path
# If not, activate your virtual environment:
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**ImportError: PyPDF2 is required for PDF support**
```bash
pip install PyPDF2
```

**LLM API Key Issues**
```bash
# Set your API key (get one at https://platform.openai.com)
export GOOGLE_API_KEY="AIzaSyB1-..."  # macOS/Linux
# Or on Windows:
set GOOGLE_API_KEY=AIzaSyB1-...
```

**Large PDF Processing**
```python
# For very large PDFs, use smaller chunk sizes
encoder = MemvidEncoder()
encoder.add_pdf("large_book.pdf", chunk_size=400, overlap=50)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=memvid tests/

# Format code
black memvid/
```

## 🆚 Comparison with Traditional Solutions

| Feature | Memvid | Vector DBs | Traditional DBs |
|---------|--------|------------|-----------------|
| Storage Efficiency | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| Setup Complexity | Simple | Complex | Complex |
| Semantic Search | ✅ | ✅ | ❌ |
| Offline Usage | ✅ | ❌ | ✅ |
| Portability | File-based | Server-based | Server-based |
| Scalability | Millions | Millions | Billions |
| Cost | Free | $$$$ | $$$ |


## 📚 Examples

Check out the [examples/](examples/) directory for:
- Building memory from Wikipedia dumps
- Creating a personal knowledge base
- Multi-language support
- Real-time memory updates
- Integration with popular LLMs

## 🆘 Getting Help

- 📖 [Documentation](https://github.com/olow304/memvid/wiki) - Comprehensive guides
- 💬 [Discussions](https://github.com/olow304/memvid/discussions) - Ask questions
- 🐛 [Issue Tracker](https://github.com/olow304/memvid/issues) - Report bugs
- 🌟 [Show & Tell](https://github.com/olow304/memvid/discussions/categories/show-and-tell) - Share your projects

## 🔗 Links

- [GitHub Repository](https://github.com/olow304/memvid)
- [PyPI Package](https://pypi.org/project/memvid)
- [Changelog](https://github.com/olow304/memvid/releases)


## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Created by [Olow304](https://github.com/olow304) and the Memvid community.

Built with ❤️ using:
- [sentence-transformers](https://www.sbert.net/) - State-of-the-art embeddings for semantic search
- [OpenCV](https://opencv.org/) - Computer vision and video processing
- [qrcode](https://github.com/lincolnloop/python-qrcode) - QR code generation
- [FAISS](https://github.com/facebookresearch/faiss) - Efficient similarity search
- [PyPDF2](https://github.com/py-pdf/pypdf) - PDF text extraction

Special thanks to all contributors who help make Memvid better!

---

**Ready to revolutionize your AI memory management? Install Memvid and start building!** 🚀



================================================
FILE: examples/book_chat.py
================================================
#!/usr/bin/env python3
"""
Book memory example using chat_with_memory
"""

import sys
import os

from memvid.config import VIDEO_FILE_TYPE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()


from memvid import MemvidEncoder, chat_with_memory

# Book PDF path - Memvid will handle PDF parsing automatically
book_pdf = "data/bitcoin.pdf"  # Replace with your PDF path

# Build memory video from PDF
video_path = f"output/book_memory.{VIDEO_FILE_TYPE}"
index_path = "output/book_memory_index.json"

# Create output directory with subdirectory for sessions
os.makedirs("output/book_chat", exist_ok=True)

# Encode PDF to video - Memvid handles all PDF parsing internally
encoder = MemvidEncoder()
encoder.add_pdf(book_pdf)  # Simple one-liner to add PDF content
encoder.build_video(video_path, index_path)
print(f"Created book memory video: {video_path}")

# Get API key from environment or use your own
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("\nNote: Set OPENAI_API_KEY environment variable for full LLM responses.")
    print("Without it, you'll only see raw context chunks.\n")

# Chat with the book - interactive session
print("\n📚 Chat with your book! Ask questions about the content.")
print("Example questions:")
print("- 'What is this document about?'")
print("- 'What are the key concepts explained?'\n")

chat_with_memory(video_path, index_path, api_key=api_key, session_dir="output/book_chat")


================================================
FILE: examples/build_memory.py
================================================
#!/usr/bin/env python3
"""
Example: Create video memory and index from text data
"""

import sys
import os

from memvid.config import VIDEO_FILE_TYPE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memvid import MemvidEncoder
import time


def main():
    # Example data - could be from files, databases, etc.
    chunks = [
        "The quantum computer achieved 100 qubits of processing power in March 2024.",
        "Machine learning models can now process over 1 trillion parameters efficiently.",
        "The new GPU architecture delivers 5x performance improvement for AI workloads.",
        "Cloud storage costs have decreased by 80% over the past five years.",
        "Quantum encryption methods are becoming standard for secure communications.",
        "Edge computing reduces latency to under 1ms for critical applications.",
        "Neural networks can now generate photorealistic images in real-time.",
        "Blockchain technology processes over 100,000 transactions per second.",
        "5G networks provide speeds up to 10 Gbps in urban areas.",
        "Autonomous vehicles have logged over 50 million miles of testing.",
        "Natural language processing accuracy has reached 98% for major languages.",
        "Robotic process automation saves companies millions in operational costs.",
        "Augmented reality glasses now have 8-hour battery life.",
        "Biometric authentication systems have false positive rates below 0.001%.",
        "Distributed computing networks utilize idle resources from millions of devices.",
        "Green data centers run entirely on renewable energy sources.",
        "AI assistants can understand context across multiple conversation turns.",
        "Cybersecurity AI detects threats 50x faster than traditional methods.",
        "Digital twins simulate entire cities for urban planning.",
        "Voice cloning technology requires only 3 seconds of audio sample.",
    ]
    
    print("Memvid Example: Building Video Memory")
    print("=" * 50)
    
    # Create encoder
    encoder = MemvidEncoder()
    
    # Add chunks
    print(f"\nAdding {len(chunks)} chunks to encoder...")
    encoder.add_chunks(chunks)
    
    # You can also add from text with automatic chunking
    additional_text = """
    The future of computing lies in the convergence of multiple technologies.
    Quantum computing will solve problems that are intractable for classical computers.
    AI and machine learning will become embedded in every application.
    The edge and cloud will work together seamlessly to process data where it makes most sense.
    Privacy-preserving technologies will enable collaboration without exposing sensitive data.
    """
    
    print("\nAdding additional text with automatic chunking...")
    encoder.add_text(additional_text, chunk_size=100, overlap=20)
    
    # Get stats
    stats = encoder.get_stats()
    print(f"\nEncoder stats:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Total characters: {stats['total_characters']}")
    print(f"  Average chunk size: {stats['avg_chunk_size']:.1f} chars")
    
    # Build video and index
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    video_file = os.path.join(output_dir, f"memory.{VIDEO_FILE_TYPE}]")
    index_file = os.path.join(output_dir, "memory_index.json")
    
    print(f"\nBuilding video: {video_file}")
    print(f"Building index: {index_file}")
    
    start_time = time.time()
    build_stats = encoder.build_video(video_file, index_file, show_progress=True)
    elapsed = time.time() - start_time
    
    print(f"\nBuild completed in {elapsed:.2f} seconds")
    print(f"\nVideo stats:")
    print(f"  Duration: {build_stats['duration_seconds']:.1f} seconds")
    print(f"  Size: {build_stats['video_size_mb']:.2f} MB")
    print(f"  FPS: {build_stats['fps']}")
    print(f"  Chunks per second: {build_stats['total_chunks'] / elapsed:.1f}")
    
    print("\nIndex stats:")
    for key, value in build_stats['index_stats'].items():
        print(f"  {key}: {value}")
    
    print("\nSuccess! Video memory created.")
    print(f"\nYou can now use this memory with:")
    print(f"  python examples/chat_memory.py")


if __name__ == "__main__":
    main()


================================================
FILE: examples/chat_memory.py
================================================
#!/usr/bin/env python3
"""
Example: Interactive conversation using MemvidChat
"""

import sys
import os

from memvid.config import VIDEO_FILE_TYPE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memvid import MemvidChat
import time


def print_search_results(results):
    """Pretty print search results"""
    print("\nRelevant context found:")
    print("-" * 50)
    for i, result in enumerate(results[:3]):
        print(f"\n[{i+1}] Score: {result['score']:.3f}")
        print(f"Text: {result['text'][:150]}...")
        print(f"Frame: {result['frame']}")


def main():
    print("Memvid Example: Interactive Chat with Memory")
    print("=" * 50)
    
    # Check if memory files exist
    video_file = f"output/memory.{VIDEO_FILE_TYPE}"
    index_file = "output/memory_index.json"
    
    if not os.path.exists(video_file) or not os.path.exists(index_file):
        print("\nError: Memory files not found!")
        print("Please run 'python examples/build_memory.py' first to create the memory.")
        return
    
    # Initialize chat
    print(f"\nLoading memory from: {video_file}")
    
    # You can set OPENAI_API_KEY, GOOGLE_API_KEY or ANTHROPIC_API_KEY as an environment variable or pass it here
    api_key = "your-api-key-here"
    if not api_key:
        print("\nNote: No OpenAI API key found. Chat will work in context-only mode.")
        print("Set OPENAI_API_KEY environment variable to enable full chat capabilities.")
    
    chat = MemvidChat(video_file, index_file, llm_api_key=api_key)
    chat.start_session()
    
    # Get stats
    stats = chat.get_stats()
    print(f"\nMemory loaded successfully!")
    print(f"  Total chunks: {stats['retriever_stats']['index_stats']['total_chunks']}")
    print(f"  LLM available: {stats['llm_available']}")
    if stats['llm_available']:
        print(f"  LLM model: {stats['llm_model']}")
    
    print("\nInstructions:")
    print("- Type your questions to search the memory")
    print("- Type 'search <query>' to see raw search results")
    print("- Type 'stats' to see system statistics")
    print("- Type 'export' to save conversation")
    print("- Type 'exit' or 'quit' to end the session")
    print("-" * 50)
    
    # Interactive loop
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
                
            # Handle commands
            if user_input.lower() in ['exit', 'quit']:
                print("\nGoodbye!")
                break
                
            elif user_input.lower() == 'stats':
                stats = chat.get_stats()
                print("\nSystem Statistics:")
                print(f"  Messages: {stats['message_count']}")
                print(f"  Cache size: {stats['retriever_stats']['cache_size']}")
                print(f"  Video frames: {stats['retriever_stats']['total_frames']}")
                continue
                
            elif user_input.lower() == 'export':
                export_file = f"output/session_{chat.session_id}.json"
                chat.export_session(export_file)
                print(f"Session exported to: {export_file}")
                continue
                
            elif user_input.lower().startswith('search '):
                query = user_input[7:]
                print(f"\nSearching for: '{query}'")
                start_time = time.time()
                results = chat.search_context(query, top_k=5)
                elapsed = time.time() - start_time
                print(f"Search completed in {elapsed:.3f} seconds")
                print_search_results(results)
                continue
            
            # Regular chat
            print("\nAssistant: ", end="", flush=True)
            start_time = time.time()
            response = chat.chat(user_input)
            elapsed = time.time() - start_time
            
            print(response)
            print(f"\n[Response time: {elapsed:.2f}s]")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue
    
    # Export session on exit
    if chat.get_history():
        export_file = f"output/session_{chat.session_id}.json"
        chat.export_session(export_file)
        print(f"\nSession saved to: {export_file}")


if __name__ == "__main__":
    main()


================================================
FILE: examples/chat_memory_fixed.py
================================================
#!/usr/bin/env python3
"""
Example: Interactive conversation using MemvidChat (Fixed)
"""

import sys
import os

from memvid.config import VIDEO_FILE_TYPE

# Set environment variable before importing transformers
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memvid import MemvidChat
import time


def print_search_results(results):
    """Pretty print search results"""
    print("\nRelevant context found:")
    print("-" * 50)
    for i, result in enumerate(results[:3]):
        print(f"\n[{i+1}] Score: {result['score']:.3f}")
        print(f"Text: {result['text'][:150]}...")
        print(f"Frame: {result['frame']}")


def main():
    print("Memvid Example: Interactive Chat with Memory")
    print("=" * 50)
    
    # Check if memory files exist
    video_file = F"output/memory.{VIDEO_FILE_TYPE}"
    index_file = "output/memory_index.json"
    
    if not os.path.exists(video_file) or not os.path.exists(index_file):
        print("\nError: Memory files not found!")
        print("Please run 'python examples/build_memory.py' first to create the memory.")
        return
    
    # Initialize chat
    print(f"\nLoading memory from: {video_file}")
    
    # You can set OPENAI_API_KEY environment variable or pass it here
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("\nNote: No OpenAI API key found. Chat will work in context-only mode.")
        print("Set OPENAI_API_KEY environment variable to enable full chat capabilities.")
    
    chat = MemvidChat(video_file, index_file, llm_api_key=api_key)
    chat.start_session()
    
    # Get stats
    stats = chat.get_stats()
    print(f"\nMemory loaded successfully!")
    print(f"  Total chunks: {stats['retriever_stats']['index_stats']['total_chunks']}")
    print(f"  LLM available: {stats['llm_available']}")
    if stats['llm_available']:
        print(f"  LLM model: {stats['llm_model']}")
    
    print("\nInstructions:")
    print("- Type your questions to search the memory")
    print("- Type 'search <query>' to see raw search results")
    print("- Type 'stats' to see system statistics")
    print("- Type 'export' to save conversation")
    print("- Type 'exit' or 'quit' to end the session")
    print("-" * 50)
    
    # Interactive loop
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
                
            # Handle commands
            if user_input.lower() in ['exit', 'quit']:
                print("\nGoodbye!")
                break
                
            elif user_input.lower() == 'stats':
                stats = chat.get_stats()
                print("\nSystem Statistics:")
                print(f"  Messages: {stats['message_count']}")
                print(f"  Cache size: {stats['retriever_stats']['cache_size']}")
                print(f"  Video frames: {stats['retriever_stats']['total_frames']}")
                continue
                
            elif user_input.lower() == 'export':
                export_file = f"output/session_{chat.session_id}.json"
                chat.export_session(export_file)
                print(f"Session exported to: {export_file}")
                continue
                
            elif user_input.lower().startswith('search '):
                query = user_input[7:]
                print(f"\nSearching for: '{query}'")
                start_time = time.time()
                results = chat.search_context(query, top_k=5)
                elapsed = time.time() - start_time
                print(f"Search completed in {elapsed:.3f} seconds")
                print_search_results(results)
                continue
            
            # Regular chat
            print("\nAssistant: ", end="", flush=True)
            start_time = time.time()
            response = chat.chat(user_input)
            elapsed = time.time() - start_time
            
            print(response)
            print(f"\n[Response time: {elapsed:.2f}s]")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue
    
    # Export session on exit
    if chat.get_history():
        export_file = f"output/session_{chat.session_id}.json"
        chat.export_session(export_file)
        print(f"\nSession saved to: {export_file}")


if __name__ == "__main__":
    main()


================================================
FILE: examples/codec_comparison.py
================================================
#!/usr/bin/env python3
"""
Multi-Codec Comparison Tool - Compare ALL available codecs on YOUR data
"""

import sys
import argparse
import time
from pathlib import Path
import json

# Add project root to Python path - works from anywhere
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Also try adding current directory if script is in project root
sys.path.insert(0, str(Path.cwd()))

try:
    from memvid.encoder import MemvidEncoder
    from memvid.config import codec_parameters, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
    print(f"✅ Imported MemvidEncoder from: {project_root}")
except ImportError as e:
    print(f"❌ Could not import MemvidEncoder from {project_root}")
    print(f"   Error: {e}")
    print(f"   Current working directory: {Path.cwd()}")
    print(f"   Python path includes:")
    for p in sys.path[:5]:  # Show first 5 paths
        print(f"     {p}")
    print()
    print("Solutions:")
    print("1. Run from project root: cd memvid && python examples/codec_comparison.py ...")
    print("2. Install package: pip install -e .")
    print("3. Set PYTHONPATH: export PYTHONPATH=/path/to/memvid:$PYTHONPATH")
    sys.exit(1)


def format_size(bytes_size):
    """Format file size in human readable format."""
    if bytes_size == 0:
        return "0 B"
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"

def get_available_codecs(encoder):
    """Get list of available codecs and their backends - FIXED"""
    # Import the full codec mapping, not just the default config
    from memvid.config import codec_parameters

    available_codecs = {}

    for codec in codec_parameters.keys():
        if codec == "mp4v":
            available_codecs[codec] = "native"
        else:
            if encoder.dcker_mngr and encoder.dcker_mngr.should_use_docker(codec):
                available_codecs[codec] = "docker"
            else:
                available_codecs[codec] = "native_ffmpeg"

    return available_codecs

def load_user_data(input_path, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP):
    """Load data from user's file or directory - FIXED VERSION"""
    input_path = Path(input_path)

    if not input_path.exists():
        print(f"❌ Path not found: {input_path}")
        return None, None

    encoder = MemvidEncoder()

    # Handle directory vs single file
    if input_path.is_dir():
        print(f"📂 Loading directory: {input_path}")

        # Find supported files in directory
        supported_extensions = ['.pdf', '.epub', '.txt', '.md', '.json']
        files = []

        for ext in supported_extensions:
            found = list(input_path.rglob(f"*{ext}"))
            files.extend(found)

        if not files:
            print(f"❌ No supported files found in {input_path}")
            print(f"   Looking for: {', '.join(supported_extensions)}")
            return None, None

        print(f"📁 Found {len(files)} supported files")

        # Load all files
        total_files_processed = 0
        total_files_failed = 0

        for file_path in files:
            try:
                if file_path.suffix.lower() == '.pdf':
                    encoder.add_pdf(str(file_path), chunk_size=chunk_size, overlap=overlap)
                elif file_path.suffix.lower() == '.epub':
                    encoder.add_epub(str(file_path), chunk_size=chunk_size, overlap=overlap)
                elif file_path.suffix.lower() == '.json':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        chunks = json.load(f)
                    if isinstance(chunks, list):
                        encoder.add_chunks(chunks)
                else:
                    # Text file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    encoder.add_text(text, chunk_size=chunk_size, overlap=overlap)

                total_files_processed += 1
                print(f"   ✅ Processed: {file_path.name}")

            except Exception as e:
                print(f"   ❌ Failed to process {file_path.name}: {e}")
                total_files_failed += 1

        if not encoder.chunks:
            print("❌ No content extracted from any files")
            return None, None

        total_chars = sum(len(chunk) for chunk in encoder.chunks)

        info = {
            'type': 'directory',
            'path': str(input_path),
            'files_processed': total_files_processed,
            'files_failed': total_files_failed,
            'chunks': len(encoder.chunks),
            'total_chars': total_chars,
            'avg_chunk_size': total_chars / len(encoder.chunks)
        }

        print(f"📊 Summary: {total_files_processed} files processed, {len(encoder.chunks)} chunks extracted")

        return encoder, info

    else:
        # Single file - your existing logic
        print(f"📄 Loading file: {input_path.name}")

        try:
            if input_path.suffix.lower() == '.pdf':
                encoder.add_pdf(str(input_path), chunk_size=chunk_size, overlap=overlap)
            elif input_path.suffix.lower() == '.epub':
                encoder.add_epub(str(input_path), chunk_size=chunk_size, overlap=overlap)
            elif input_path.suffix.lower() == '.json':
                with open(input_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                if isinstance(chunks, list):
                    encoder.add_chunks(chunks)
                else:
                    print("❌ JSON file must contain a list of text chunks")
                    return None, None
            else:
                with open(input_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                encoder.add_text(text, chunk_size=chunk_size, overlap=overlap)
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return None, None

        if not encoder.chunks:
            print("❌ No content extracted from file")
            return None, None

        total_chars = sum(len(chunk) for chunk in encoder.chunks)

        info = {
            'type': 'file',
            'file_name': input_path.name,
            'file_size': input_path.stat().st_size,
            'chunks': len(encoder.chunks),
            'total_chars': total_chars,
            'avg_chunk_size': total_chars / len(encoder.chunks)
        }

        return encoder, info

def test_codec(encoder, codec, name_stem, output_dir):
    """Test encoding with a specific codec - FIXED"""
    print(f"\n🎬 Testing {codec.upper()} encoding...")

    # Create fresh encoder copy to avoid state issues
    test_encoder = MemvidEncoder()
    test_encoder.chunks = encoder.chunks.copy()

    # Determine file extension from full codec mapping - FIXED
    file_ext = codec_parameters[codec]["video_file_type"]

    # FIX: Add missing dot before extension
    output_path = output_dir / f"{name_stem}_{codec}.{file_ext}"
    index_path = output_dir / f"{name_stem}_{codec}.json"

    start_time = time.time()

    try:
        stats = test_encoder.build_video(
            str(output_path),
            str(index_path),
            codec=codec,
            show_progress=False,  # Keep output clean
            auto_build_docker=True,
            allow_fallback=False  # Fail rather than fallback for comparison
        )

        encoding_time = time.time() - start_time

        # Check if file was actually created and has content
        if output_path.exists():
            file_size = output_path.stat().st_size
        else:
            print(f"   ❌ Output file was not created: {output_path}")
            return {
                'success': False,
                'codec': codec,
                'error': "Output file was not created"
            }

        # Check for zero-byte files
        if file_size == 0:
            print(f"   ❌ Zero-byte file created - encoding failed")
            return {
                'success': False,
                'codec': codec,
                'error': "Zero-byte file created"
            }

        result = {
            'success': True,
            'codec': codec,
            'backend': stats.get('backend', 'unknown'),
            'file_size': file_size,
            'file_size_mb': file_size / (1024 * 1024),
            'encoding_time': encoding_time,
            'chunks_per_mb': len(encoder.chunks) / (file_size / (1024 * 1024)) if file_size > 0 else 0,
            'path': output_path,
            'file_ext': file_ext
        }

        print(f"   ✅ Success: {format_size(file_size)} in {encoding_time:.1f}s via {result['backend']}")
        return result

    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return {
            'success': False,
            'codec': codec,
            'error': str(e)
        }

def run_multi_codec_comparison(encoder, data_info, codecs, output_dir="output"):
    """Run comparison across multiple codecs - FIXED for directories"""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Generate output name based on input type - FIXED
    timestamp = int(time.time())

    if data_info['type'] == 'directory':
        # Use directory name for collections
        dir_name = Path(data_info['path']).name
        name_stem = f"{dir_name}_{data_info['files_processed']}files_{timestamp}"
    else:
        # Use file name for single files
        name_stem = f"{Path(data_info['file_name']).stem}_{timestamp}"

    print(f"\n🏁 Multi-Codec Comparison Starting")
    print("=" * 60)
    print(f"📁 Output prefix: {name_stem}")

    results = {}

    for codec in codecs:
        results[codec] = test_codec(encoder, codec, name_stem, output_path)

    return results

def print_comparison_table(data_info, results, codecs):
    """Print detailed comparison table - FIXED zero division"""

    print(f"\n📊 MULTI-CODEC COMPARISON RESULTS")
    print("=" * 80)

    if data_info['type'] == 'directory':
        print(f"📁 Source: {data_info['path']} ({data_info['files_processed']} files)")
    else:
        print(f"📄 Source: {data_info['file_name']}")

    print(f"📋 Content: {data_info['chunks']} chunks, {format_size(data_info['total_chars'])} characters")
    print()

    # Prepare table data
    successful_results = [(codec, result) for codec, result in results.items() if result['success']]
    failed_results = [(codec, result) for codec, result in results.items() if not result['success']]

    if successful_results:
        print("✅ SUCCESSFUL ENCODINGS:")
        print("-" * 80)
        print(f"{'Codec':<8} {'Backend':<12} {'Size':<12} {'Chunks/MB':<10} {'Time':<8} {'Ratio':<8}")
        print("-" * 80)

        # Sort by file size (smallest first)
        successful_results.sort(key=lambda x: x[1]['file_size'])

        # FIX: Handle zero baseline size
        baseline_size = successful_results[0][1]['file_size'] if successful_results else 1
        if baseline_size == 0:
            baseline_size = 1  # Avoid division by zero

        for codec, result in successful_results:
            size_str = format_size(result['file_size'])
            chunks_per_mb = f"{result['chunks_per_mb']:.0f}" if result['chunks_per_mb'] > 0 else "N/A"
            time_str = f"{result['encoding_time']:.1f}s"
            backend = result['backend']
            ratio = result['file_size'] / baseline_size if baseline_size > 0 else 1.0
            ratio_str = f"{ratio:.1f}x" if ratio != 1.0 else "baseline"

            print(f"{codec:<8} {backend:<12} {size_str:<12} {chunks_per_mb:<10} {time_str:<8} {ratio_str:<8}")

        # Find the best compression ratio and best speed - with safety checks
        if successful_results:
            best_compression = min(successful_results, key=lambda x: x[1]['file_size'])
            fastest = min(successful_results, key=lambda x: x[1]['encoding_time'])

            print()
            print(f"🏆 Best Compression: {best_compression[0].upper()} ({format_size(best_compression[1]['file_size'])})")
            print(f"⚡ Fastest Encoding: {fastest[0].upper()} ({fastest[1]['encoding_time']:.1f}s)")

            # Calculate storage efficiency
            total_chunks = data_info['chunks']
            print(f"\n💾 Storage Efficiency (chunks per MB):")
            print(f"📦 Total chunks in dataset: {total_chunks}")

            storage_efficiency = []
            for codec, result in successful_results:
                # Get file size in MB
                file_size_mb = result['file_size'] / (1024 * 1024)  # Convert bytes to MB

                # Calculate chunks per MB
                chunks_per_mb = total_chunks / file_size_mb if file_size_mb > 0 else 0

                storage_efficiency.append((codec, chunks_per_mb, file_size_mb))
                print(f"   {codec.upper()}: {chunks_per_mb:.1f} chunks/MB ({file_size_mb:.1f} MB total)")

            # Sort by efficiency (highest chunks per MB first)
            storage_efficiency.sort(key=lambda x: x[1], reverse=True)

            print(f"\n🏆 Storage Efficiency Ranking:")
            for i, (codec, chunks_per_mb, file_size_mb) in enumerate(storage_efficiency, 1):
                efficiency_vs_best = chunks_per_mb / storage_efficiency[0][1] if storage_efficiency[0][1] > 0 else 0
                print(f"   {i}. {codec.upper()}: {chunks_per_mb:.1f} chunks/MB ({efficiency_vs_best:.1%} of best)")

    if failed_results:
        print(f"\n❌ FAILED ENCODINGS:")
        print("-" * 40)
        for codec, result in failed_results:
            print(f"   {codec.upper()}: {result['error']}")

    print(f"\n📁 Output files saved to: {Path('output').absolute()}")

def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple video codecs on YOUR data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('input_path', help='Path to your file (PDF, EPUB, TXT, JSON)')
    parser.add_argument('--codecs', nargs='+', default=['mp4v', 'h265'],
                        help='Codecs to test (default: mp4v h265). Use "all" for all available.')
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE,
                        help='Chunk size for text splitting')
    parser.add_argument('--overlap', type=int, default=DEFAULT_OVERLAP,
                        help='Overlap for text splitting')
    parser.add_argument('--output-dir', default='output',
                        help='Output directory (default: output)')

    args = parser.parse_args()

    print("🎥 Memvid Multi-Codec Comparison Tool")
    print("=" * 50)

    # Load user data
    encoder, data_info = load_user_data(args.input_path, args.chunk_size)
    if not encoder:
        sys.exit(1)

    # Determine available codecs
    available_codecs = get_available_codecs(encoder)

    print(f"\n🎛️  Available Codecs:")
    for codec, backend in available_codecs.items():
        print(f"   {codec.upper()}: {backend}")

    # Parse codec selection
    if args.codecs == ['all']:
        test_codecs = list(available_codecs.keys())
    else:
        test_codecs = []
        for codec in args.codecs:
            if codec in available_codecs:
                test_codecs.append(codec)
            else:
                print(f"⚠️  Codec '{codec}' not available, skipping")

        if not test_codecs:
            print("❌ No valid codecs specified")
            sys.exit(1)

    print(f"\n🧪 Testing Codecs: {', '.join(test_codecs)}")

    # Show Docker status
    docker_status = encoder.get_docker_status()
    print(f"🐳 Docker Status: {docker_status}")

    # Run comparison
    results = run_multi_codec_comparison(encoder, data_info, test_codecs, args.output_dir)

    # Show results
    print_comparison_table(data_info, results, test_codecs)

    print(f"\n🎉 Multi-codec comparison complete!")

if __name__ == '__main__':
    main()


================================================
FILE: examples/file_chat.py
================================================
#!/usr/bin/env python3
"""
file_chat.py - Enhanced script for testing MemvidChat with external files

This script allows you to:
1. Create a memory video from your own files with configurable parameters
2. Chat with the created memory using different LLM providers
3. Store results in output/ directory to avoid contaminating the main repo
4. Handle FAISS training issues gracefully
5. Configure chunking and compression parameters

Usage:
    python file_chat.py --input-dir /path/to/documents --provider google
    python file_chat.py --files file1.txt file2.pdf --provider openai --chunk-size 2048
    python file_chat.py --load-existing output/my_memory --provider google
    python file_chat.py --input-dir ~/docs --index-type Flat --codec h265

Examples:
    # Create memory from a directory and chat with Google
    python file_chat.py --input-dir ~/Documents/research --provider google

    # Create memory with custom chunking for large documents
    python file_chat.py --files report.pdf --chunk-size 2048 --overlap 32 --provider openai

    # Use Flat index for small datasets (avoids FAISS training issues)
    python file_chat.py --files single_doc.pdf --index-type Flat --provider google

    # Load existing memory and continue chatting
    python file_chat.py --load-existing output/research_memory --provider google

    # Create memory with H.265 compression
    python file_chat.py --input-dir ~/docs --codec h265 --provider anthropic
"""

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime
import json

# Add the parent directory to the path so we can import memvid
sys.path.insert(0, str(Path(__file__).parent.parent))  # Go up TWO levels from examples/

from memvid import MemvidEncoder, MemvidChat
from memvid.config import get_default_config, get_codec_parameters

def setup_output_dir():
    """Create output directory if it doesn't exist"""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    return output_dir

def generate_memory_name(input_source):
    """Generate a meaningful name for the memory files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if isinstance(input_source, list):
        # Multiple files
        base_name = f"files_{len(input_source)}items"
    else:
        # Directory
        dir_name = Path(input_source).name
        base_name = f"dir_{dir_name}"

    return f"{base_name}_{timestamp}"

def collect_files_from_directory(directory_path, extensions=None):
    """Collect supported files from a directory"""
    if extensions is None:
        extensions = {'.txt', '.md', '.pdf', '.doc', '.docx', '.rtf', '.epub', '.html', '.htm'}

    directory = Path(directory_path)
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory_path}")

    files = []
    for ext in extensions:
        files.extend(directory.rglob(f"*{ext}"))

    return [str(f) for f in files if f.is_file()]

def create_memory_with_fallback(encoder, video_path, index_path):
    """Create memory with graceful FAISS fallback for training issues"""
    try:
        build_stats = encoder.build_video(str(video_path), str(index_path))
        return build_stats
    except Exception as e:
        error_str = str(e)
        if "is_trained" in error_str or "IndexIVFFlat" in error_str or "training" in error_str.lower():
            print(f"⚠️  FAISS IVF training failed: {e}")
            print(f"🔄 Auto-switching to Flat index for compatibility...")

            # Override config to use Flat index
            original_index_type = encoder.config["index"]["type"]
            encoder.config["index"]["type"] = "Flat"

            try:
                # Recreate the index manager with Flat index
                encoder._setup_index()
                build_stats = encoder.build_video(str(video_path), str(index_path))
                print(f"✅ Successfully created memory using Flat index")
                return build_stats
            except Exception as fallback_error:
                print(f"❌ Fallback also failed: {fallback_error}")
                raise
        else:
            raise

def create_memory_from_files(files, output_dir, memory_name, **config_overrides):
    """Create a memory video from a list of files with configurable parameters"""
    print(f"Creating memory from {len(files)} files...")

    # Start timing
    start_time = time.time()

    # Apply config overrides to default config
    config = get_default_config()
    for key, value in config_overrides.items():
        if key in ['chunk_size', 'overlap']:
            config["chunking"][key] = value
        elif key == 'index_type':
            config["index"]["type"] = value
        elif key == 'codec':
            config[key] = value

    # Initialize encoder with config first (this ensures config consistency)
    encoder = MemvidEncoder(config)

    # Get the actual codec and video extension from the encoder's config
    actual_codec = encoder.config.get("codec")  # Use encoder's resolved codec
    video_ext = get_codec_parameters(actual_codec).get("video_file_type", "mp4")

    # Import tqdm for progress bars
    try:
        from tqdm import tqdm
        use_progress = True
    except ImportError:
        print("Note: Install tqdm for progress bars (pip install tqdm)")
        use_progress = False

    processed_count = 0
    skipped_count = 0

    # Process files with progress tracking
    file_iterator = tqdm(files, desc="Processing files") if use_progress else files

    for file_path in file_iterator:
        file_path = Path(file_path)
        if not use_progress:
            print(f"Processing: {file_path.name}")

        try:
            chunk_size = config["chunking"]["chunk_size"]
            overlap = config["chunking"]["overlap"]

            if file_path.suffix.lower() == '.pdf':
                encoder.add_pdf(str(file_path), chunk_size, overlap)
            elif file_path.suffix.lower() == '.epub':
                encoder.add_epub(str(file_path), chunk_size, overlap)
            elif file_path.suffix.lower() in ['.html', '.htm']:
                # Process HTML with BeautifulSoup
                try:
                    from bs4 import BeautifulSoup
                except ImportError:
                    print(f"Warning: BeautifulSoup not available for HTML processing. Skipping {file_path.name}")
                    skipped_count += 1
                    continue

                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    soup = BeautifulSoup(f.read(), 'html.parser')
                    for script in soup(["script", "style"]):
                        script.decompose()
                    text = soup.get_text()
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    clean_text = ' '.join(chunk for chunk in chunks if chunk)
                    if clean_text.strip():
                        encoder.add_text(clean_text, chunk_size, overlap)
            else:
                # Read as text file
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if content.strip():
                        encoder.add_text(content, chunk_size, overlap)

            processed_count += 1

        except Exception as e:
            print(f"Warning: Could not process {file_path.name}: {e}")
            skipped_count += 1
            continue

    processing_time = time.time() - start_time
    print(f"\n📊 Processing Summary:")
    print(f"  ✅ Successfully processed: {processed_count} files")
    print(f"  ⚠️  Skipped: {skipped_count} files")
    print(f"  ⏱️  Processing time: {processing_time:.2f} seconds")

    if processed_count == 0:
        raise ValueError("No files were successfully processed")

    # Build the video (video_ext already determined from encoder config)
    video_path = output_dir / f"{memory_name}.{video_ext}"
    index_path = output_dir / f"{memory_name}_index.json"

    print(f"\n🎬 Building memory video: {video_path}")
    print(f"📊 Total chunks to encode: {len(encoder.chunks)}")

    encoding_start = time.time()

    # Use fallback-enabled build function
    build_stats = create_memory_with_fallback(encoder, video_path, index_path)

    encoding_time = time.time() - encoding_start
    total_time = time.time() - start_time

    # Enhanced statistics
    print(f"\n🎉 Memory created successfully!")
    print(f"  📁 Video: {video_path}")
    print(f"  📋 Index: {index_path}")
    print(f"  📊 Chunks: {build_stats.get('total_chunks', 'unknown')}")
    print(f"  🎞️  Frames: {build_stats.get('total_frames', 'unknown')}")
    print(f"  📏 Video size: {video_path.stat().st_size / (1024 * 1024):.1f} MB")
    print(f"  ⏱️  Encoding time: {encoding_time:.2f} seconds")
    print(f"  ⏱️  Total time: {total_time:.2f} seconds")

    if build_stats.get('video_size_mb', 0) > 0:
        # Calculate rough compression stats
        total_chars = sum(len(chunk) for chunk in encoder.chunks)
        original_size_mb = total_chars / (1024 * 1024)  # Rough estimate
        compression_ratio = original_size_mb / build_stats['video_size_mb'] if build_stats['video_size_mb'] > 0 else 0
        print(f"  📦 Estimated compression ratio: {compression_ratio:.1f}x")

    # Save metadata about this memory
    metadata = {
        'created': datetime.now().isoformat(),
        'source_files': files,
        'video_path': str(video_path),
        'index_path': str(index_path),
        'config_used': config,
        'processing_stats': {
            'files_processed': processed_count,
            'files_skipped': skipped_count,
            'processing_time_seconds': processing_time,
            'encoding_time_seconds': encoding_time,
            'total_time_seconds': total_time
        },
        'build_stats': build_stats
    }

    metadata_path = output_dir / f"{memory_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  📄 Metadata: {metadata_path}")

    return str(video_path), str(index_path)

def load_existing_memory(memory_path):
    """Load and validate existing memory from the output directory"""
    memory_path = Path(memory_path)

    # Handle different input formats
    if memory_path.is_dir():
        # Directory provided, look for memory files
        # Try all possible video extensions
        video_files = []
        for ext in ['mp4', 'avi', 'mkv']:
            video_files.extend(memory_path.glob(f"*.{ext}"))

        if not video_files:
            raise ValueError(f"No video files found in {memory_path}")

        video_path = video_files[0]
        # Look for corresponding index file
        possible_index_paths = [
            video_path.with_name(video_path.stem + '_index.json'),
            video_path.with_suffix('.json'),
            video_path.with_suffix('_index.json')
        ]

        index_path = None
        for possible_path in possible_index_paths:
            if possible_path.exists():
                index_path = possible_path
                break

        if not index_path:
            raise ValueError(f"No index file found for {video_path}")

    elif memory_path.suffix in ['.mp4', '.avi', '.mkv']:
        # Video file provided
        video_path = memory_path
        index_path = memory_path.with_name(memory_path.stem + '_index.json')

    else:
        # Assume it's a base name, try to find files
        base_path = memory_path
        video_path = None

        # Try different video extensions
        for ext in ['mp4', 'avi', 'mkv']:
            candidate = base_path.with_suffix(f'.{ext}')
            if candidate.exists():
                video_path = candidate
                break

        if not video_path:
            raise ValueError(f"No video file found with base name: {memory_path}")

        index_path = base_path.with_suffix('_index.json')

    # Validate files exist and are readable
    if not video_path.exists():
        raise ValueError(f"Video file not found: {video_path}")
    if not index_path.exists():
        raise ValueError(f"Index file not found: {index_path}")

    # Validate file integrity
    try:
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        chunk_count = len(index_data.get('metadata', []))
        print(f"✅ Index contains {chunk_count} chunks")
    except Exception as e:
        raise ValueError(f"Index file corrupted: {e}")

    # Check video file size
    video_size_mb = video_path.stat().st_size / (1024 * 1024)
    print(f"✅ Video file: {video_size_mb:.1f} MB")

    print(f"Loading existing memory:")
    print(f"  📁 Video: {video_path}")
    print(f"  📋 Index: {index_path}")

    return str(video_path), str(index_path)

def start_chat_session(video_path, index_path, provider='google', model=None):
    """Start an interactive chat session"""
    print(f"\nInitializing chat with {provider}...")

    try:
        chat = MemvidChat(
            video_file=video_path,
            index_file=index_path,
            llm_provider=provider,
            llm_model=model
        )

        print("✓ Chat initialized successfully!")
        print("\nStarting interactive session...")
        print("Commands:")
        print("  - Type your questions normally")
        print("  - Type 'quit' or 'exit' to end")
        print("  - Type 'clear' to clear conversation history")
        print("  - Type 'stats' to see session statistics")
        print("=" * 50)

        # Start interactive chat
        while True:
            try:
                user_input = input("\nYou: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    # Export conversation before exiting
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    export_path = Path("output") / f"conversation_{timestamp}.json"
                    chat.export_conversation(str(export_path))
                    print(f"💾 Conversation saved to: {export_path}")
                    print("Goodbye!")
                    break

                elif user_input.lower() == 'clear':
                    chat.clear_history()
                    print("🗑️ Conversation history cleared")
                    continue

                elif user_input.lower() == 'stats':
                    stats = chat.get_stats()
                    print(f"📊 Session stats: {stats}")
                    continue

                if not user_input:
                    continue

                # Get response (always stream for better UX)
                chat.chat(user_input, stream=True)

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

    except Exception as e:
        print(f"Error initializing chat: {e}")
        return False

    return True

def main():
    parser = argparse.ArgumentParser(
        description="Chat with your documents using MemVid with enhanced configuration options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input-dir',
        help='Directory containing documents to process'
    )
    input_group.add_argument(
        '--files',
        nargs='+',
        help='Specific files to process'
    )
    input_group.add_argument(
        '--load-existing',
        help='Load existing memory (provide path to video file or directory)'
    )

    # LLM options
    parser.add_argument(
        '--provider',
        choices=['openai', 'google', 'anthropic'],
        default='google',
        help='LLM provider to use (default: google)'
    )
    parser.add_argument(
        '--model',
        help='Specific model to use (uses provider defaults if not specified)'
    )

    # Memory options
    parser.add_argument(
        '--memory-name',
        help='Custom name for the memory files (auto-generated if not provided)'
    )

    # Processing configuration options
    parser.add_argument(
        '--chunk-size',
        type=int,
        help='Override default chunk size (e.g., 2048, 4096)'
    )
    parser.add_argument(
        '--overlap',
        type=int,
        help='Override default chunk overlap (e.g., 16, 32, 64)'
    )
    parser.add_argument(
        '--index-type',
        choices=['Flat', 'IVF'],
        help='FAISS index type (Flat for small datasets, IVF for large datasets)'
    )
    parser.add_argument(
        '--codec',
        choices=['h264', 'h265', 'mp4v'],
        help='Video codec to use for compression'
    )

    # File processing options
    parser.add_argument(
        '--extensions',
        nargs='+',
        default=['.txt', '.md', '.pdf', '.doc', '.docx', '.epub', '.html', '.htm'],
        help='File extensions to include when processing directories'
    )

    args = parser.parse_args()

    # Setup output directory
    output_dir = setup_output_dir()

    try:
        # Get or create memory
        if args.load_existing:
            video_path, index_path = load_existing_memory(args.load_existing)
        else:
            # Collect files
            if args.input_dir:
                files = collect_files_from_directory(args.input_dir, set(args.extensions))
                if not files:
                    print(f"No supported files found in {args.input_dir}")
                    return 1
                print(f"Found {len(files)} files to process")
                input_source = args.input_dir
            else:
                files = args.files
                for f in files:
                    if not Path(f).exists():
                        print(f"File not found: {f}")
                        return 1
                input_source = files

            # Generate memory name
            memory_name = args.memory_name or generate_memory_name(input_source)

            # Build config overrides from command line arguments
            config_overrides = {}
            if args.chunk_size:
                config_overrides['chunk_size'] = args.chunk_size
            if args.overlap:
                config_overrides['overlap'] = args.overlap
            if args.index_type:
                config_overrides['index_type'] = args.index_type
            if args.codec:
                config_overrides['codec'] = args.codec

            # Show what defaults are being used if no overrides provided
            if not config_overrides:
                default_config = get_default_config()
                print(f"📋 Using default configuration:")
                print(f"   Chunk size: {default_config['chunking']['chunk_size']}")
                print(f"   Overlap: {default_config['chunking']['overlap']}")
                print(f"   Index type: {default_config['index']['type']}")
                print(f"   Codec: {default_config.get('codec', 'h265')}")

            # Create memory with configuration
            video_path, index_path = create_memory_from_files(
                files, output_dir, memory_name, **config_overrides
            )

        # Start chat session
        success = start_chat_session(video_path, index_path, args.provider, args.model)
        return 0 if success else 1

    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())


================================================
FILE: examples/simple_chat.py
================================================
#!/usr/bin/env python3
"""
Simplified interactive chat example
"""

import sys
import os

from memvid.config import VIDEO_FILE_TYPE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memvid import chat_with_memory, quick_chat

def main():
    print("Memvid Simple Chat Examples")
    print("=" * 50)
    
    video_file = F"output/memory.{VIDEO_FILE_TYPE}"
    index_file = "output/memory_index.json"
    
    # Check if memory exists
    if not os.path.exists(video_file):
        print("\nError: Run 'python examples/build_memory.py' first!")
        return
    
    # Check for API key
    api_key = "your-api-key-here"
    if not api_key:
        print("\nNote: Set OPENAI_API_KEY environment variable for full LLM responses.")
        print("Without it, you'll only see raw context chunks.\n")
        # You can also hardcode for testing (not recommended for production):
        # api_key = "your-api-key-here"
    
    print("\n1. Quick one-off query:")
    print("-" * 30)
    response = quick_chat(video_file, index_file, "How many qubits did the quantum computer achieve?", api_key=api_key)
    print(f"Response: {response}")
    
    print("\n\n2. Interactive chat session:")
    print("-" * 30)
    print("Starting interactive mode...\n")
    
    # This single line replaces all the interactive loop code!
    chat_with_memory(video_file, index_file, api_key=api_key)

if __name__ == "__main__":
    main()


================================================
FILE: examples/soccer_chat.py
================================================
#!/usr/bin/env python3
"""
Soccer memory example using chat_with_memory
"""

import sys
import os

from memvid.config import VIDEO_FILE_TYPE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memvid import MemvidEncoder, chat_with_memory

# Soccer knowledge base
soccer_chunks = [
    "Brazil has won the most World Cups with 5 titles (1958, 1962, 1970, 1994, 2002).",
    "Argentina won the 2022 World Cup in Qatar with Lionel Messi finally achieving his dream.",
    "Pelé is the only player to win 3 World Cups and scored over 1000 career goals.",
    "Diego Maradona is famous for the 'Hand of God' and solo goal vs England in 1986.",
    "Lionel Messi is a 7-time Ballon d'Or winner who finally won the World Cup in 2022.",
    "Cristiano Ronaldo is a 5-time Ballon d'Or winner and all-time Champions League top scorer.",
    "Real Madrid has won the most Champions League titles with 14 trophies.",
    "Barcelona's tiki-taka style revolutionized modern football under Pep Guardiola.",
    "Manchester United has won 13 Premier League titles, the most in the competition.",
    "Arsenal went unbeaten throughout the 2003-04 Premier League season (The Invincibles).",
    "Leicester City achieved a fairytale 5000-1 Premier League title win in 2015-16.",
    "Liverpool's 'Miracle of Istanbul' - came back from 3-0 down vs AC Milan in 2005 Champions League final.",
    "The first World Cup was held in Uruguay in 1930, with Uruguay winning on home soil.",
    "Only 8 nations have won the World Cup: Brazil, Germany, Italy, Argentina, France, Uruguay, Spain, England.",
    "Miroslav Klose (Germany) is the World Cup's all-time top scorer with 16 goals.",
    "Camp Nou (Barcelona) is Europe's largest stadium with 99,354 capacity.",
    "Old Trafford is known as the 'Theatre of Dreams' and home to Manchester United.",
    "El Clásico between Real Madrid and Barcelona is football's biggest rivalry.",
    "The 2026 World Cup will feature 48 teams for the first time.",
    "Neymar's €222 million transfer from Barcelona to PSG in 2017 is still the world record."
]

# Build memory video
video_path = f"output/soccer_memory.{VIDEO_FILE_TYPE}"
index_path = "output/soccer_memory_index.json"

# Create output directory with subdirectory for sessions
os.makedirs("output/soccer_chat", exist_ok=True)

# Encode chunks to video
encoder = MemvidEncoder()
encoder.add_chunks(soccer_chunks)
encoder.build_video(video_path, index_path)
print(f"Created soccer memory video: {video_path}")

api_key = "your-api-key-here"
if not api_key:
    print("\nNote: Set OPENAI_API_KEY environment variable for full LLM responses.")
    print("Without it, you'll only see raw context chunks.\n")

# Chat with the memory - interactive session with custom session directory
chat_with_memory(video_path, index_path, api_key=api_key, session_dir="output/soccer_chat")


