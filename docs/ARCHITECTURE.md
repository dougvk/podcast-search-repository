# Architecture Documentation

## System Overview

The Podcast Search Repository is built on a novel architecture that uses video encoding as a compressed storage mechanism for text data, enabling lightning-fast semantic search across large conversational datasets.

## Core Architecture Principles

### 1. Video-as-Database Storage
- **Concept**: Encode transcript chunks as video frames using OpenCV
- **Benefits**: Massive compression, native similarity search, parallel processing
- **Implementation**: Based on memvid's MP4 encoding technique
- **Storage Ratio**: ~1000:1 compression vs raw text

### 2. Semantic-First Search
- **Embedding Strategy**: Sentence-transformers optimized for conversational content
- **Index Technology**: FAISS for sub-second similarity search
- **Context Preservation**: Overlapping chunks maintain conversation flow
- **Multi-modal Results**: Semantic + keyword hybrid search

### 3. Conversation-Aware Processing
- **Chunking Strategy**: Preserve speaker turns and context boundaries
- **Metadata Integration**: Episode, speaker, timestamp information
- **Cross-Episode Linking**: Topic clustering across entire dataset
- **Speaker Diarization**: Maintain speaker attribution in search results

## Technical Stack

### Core Technologies
```
Storage Layer:     OpenCV + MP4 (video encoding)
Search Engine:     FAISS (vector similarity search)
Embeddings:        sentence-transformers (local inference)
API Layer:         FastAPI (async endpoints)
Web Interface:     Vanilla HTML/JS (lightweight)
Processing:        Python 3.9+ (async/await)
```

### Dependencies
```
Primary:
- opencv-python (video processing)
- faiss-cpu (vector search)
- sentence-transformers (embeddings)
- fastapi (API framework)
- pydantic (data validation)

Secondary:
- numpy (numerical operations)
- pandas (data manipulation)
- pyyaml (configuration)
- uvicorn (ASGI server)
```

## System Components

### 1. Data Processing Pipeline (`podcast/`)

#### Transcript Processor (`transcript_processor.py`)
```python
class TranscriptProcessor:
    def chunk_conversation(self, transcript: str) -> List[Chunk]:
        # Intelligent chunking preserving speaker turns
        # Configurable overlap for context preservation
        # Metadata extraction (timestamps, speakers)
        
    def optimize_for_search(self, chunks: List[Chunk]) -> List[SearchableChunk]:
        # Conversation-aware preprocessing
        # Speaker attribution preservation
        # Topic boundary detection
```

#### Metadata Extractor (`metadata_extractor.py`)
```python
class MetadataExtractor:
    def extract_episode_info(self, file_path: str) -> EpisodeMetadata:
        # File name parsing for episode info
        # Timestamp extraction from transcripts
        # Speaker identification and normalization
        
    def enrich_chunks(self, chunks: List[Chunk], metadata: EpisodeMetadata) -> List[EnrichedChunk]:
        # Add episode context to each chunk
        # Generate searchable metadata fields
        # Create cross-episode linking data
```

### 2. Core Storage System (`core/`)

#### Video Encoder (`video_encoder.py`)
```python
class VideoEncoder:
    def encode_chunks_to_video(self, chunks: List[EnrichedChunk]) -> str:
        # Convert text chunks to video frames
        # Optimize compression settings for search speed
        # Maintain chunk-to-frame mapping
        
    def decode_video_to_chunks(self, video_path: str, frame_indices: List[int]) -> List[Chunk]:
        # Fast frame extraction for search results
        # Parallel decoding for multiple results
        # Context window expansion
```

#### Search Engine (`search_engine.py`)
```python
class SearchEngine:
    def __init__(self):
        self.faiss_index = faiss.IndexIVFFlat()
        self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def semantic_search(self, query: str, k: int = 10) -> List[SearchResult]:
        # Query embedding generation
        # FAISS similarity search
        # Result ranking and filtering
        
    def hybrid_search(self, query: str) -> List[SearchResult]:
        # Combine semantic and keyword search
        # Boost exact matches
        # Conversation context expansion
```

### 3. API Layer (`api/`)

#### Search API (`search_api.py`)
```python
@app.post("/search")
async def search_podcasts(request: SearchRequest) -> SearchResponse:
    # Query processing and validation
    # Multi-threaded search execution
    # Result formatting and metadata inclusion
    
@app.get("/episodes/{episode_id}")
async def get_episode_info(episode_id: str) -> EpisodeInfo:
    # Episode metadata retrieval
    # Full transcript access
    # Related episode suggestions
```

## Data Flow Architecture

### 1. Ingestion Pipeline
```
Raw Transcripts → Transcript Processor → Metadata Extractor → Chunk Generator → Video Encoder → FAISS Index
```

### 2. Search Pipeline
```
User Query → Query Processor → Embedding Generator → FAISS Search → Video Decoder → Result Formatter → API Response
```

### 3. Storage Hierarchy
```
Level 1: Raw transcripts (data/raw_transcripts/)
Level 2: Processed chunks (data/processed/)
Level 3: Video libraries (data/video_libraries/)
Level 4: Search indexes (data/indexes/)
```

## Performance Optimizations

### 1. Batch Processing
- **Parallel Chunk Processing**: Multi-threaded transcript parsing
- **Vectorized Embedding**: Batch embedding generation
- **Efficient Video Encoding**: OpenCV optimization flags
- **Index Building**: Incremental FAISS index updates

### 2. Search Optimizations
- **Query Caching**: LRU cache for common queries
- **Index Sharding**: Separate indexes for different time periods
- **Parallel Search**: Multi-threaded similarity search
- **Result Streaming**: Async result delivery

### 3. Memory Management
- **Lazy Loading**: Load video libraries on demand
- **Memory Mapping**: Efficient large file handling
- **Garbage Collection**: Proactive cleanup of temporary data
- **Index Compression**: Quantized vector storage

## Scalability Considerations

### Horizontal Scaling
- **Sharded Storage**: Distribute video libraries across nodes
- **Load Balancing**: Multiple API instances
- **Distributed Search**: Parallel index querying
- **Caching Layer**: Redis for frequent queries

### Vertical Scaling
- **GPU Acceleration**: FAISS GPU indices for larger datasets
- **SSD Storage**: Fast random access for video files
- **Memory Optimization**: Efficient chunk representation
- **CPU Optimization**: Multi-core processing pipelines

## Security Architecture

### Data Protection
- **Local Processing**: No external API calls for core functionality
- **Encrypted Storage**: Optional disk encryption for sensitive transcripts
- **Access Control**: Token-based API authentication
- **Audit Logging**: Comprehensive search and access logs

### Privacy Considerations
- **No Data Leakage**: All processing happens locally
- **Anonymization**: Optional speaker name anonymization
- **Retention Policies**: Configurable data cleanup
- **Consent Management**: Clear data usage policies

## Configuration Management

### Environment-Specific Settings
```yaml
# config/embeddings.yaml
model_name: "all-MiniLM-L6-v2"
embedding_dimension: 384
batch_size: 32
cache_size: 10000

# config/chunking.yaml
chunk_size: 300
overlap_size: 50
preserve_speaker_turns: true
min_chunk_length: 100

# config/search.yaml
max_results: 50
similarity_threshold: 0.7
context_window: 2
boost_exact_matches: true
```

## Monitoring and Observability

### Metrics Collection
- **Search Performance**: Query latency, result quality
- **System Health**: Memory usage, disk space, CPU utilization
- **User Behavior**: Popular queries, result click-through rates
- **Error Tracking**: Failed searches, processing errors

### Logging Strategy
- **Structured Logging**: JSON format for log aggregation
- **Log Levels**: DEBUG, INFO, WARN, ERROR, CRITICAL
- **Rotation Policy**: Daily rotation with 30-day retention
- **Search Analytics**: Query patterns and performance metrics

## Deployment Architecture

### Local Development
```bash
# Single-node deployment
docker-compose up -d
# Includes: API server, web interface, background processing
```

### Production Deployment
```bash
# Multi-container setup
docker-compose -f docker-compose.prod.yml up -d
# Includes: Load balancer, multiple API instances, monitoring
```

### Backup Strategy
- **Incremental Backups**: Daily video library backups
- **Index Snapshots**: Weekly FAISS index backups
- **Configuration Versioning**: Git-based config management
- **Disaster Recovery**: Automated restoration procedures

## Future Architecture Considerations

### Planned Enhancements
1. **Real-time Processing**: Stream new episodes as they arrive
2. **Multi-language Support**: Embeddings for non-English content
3. **Advanced Analytics**: Topic modeling and trend analysis
4. **Mobile Integration**: Optimized mobile search interface
5. **Collaborative Features**: Shared annotations and bookmarks

### Technical Debt Management
- **Code Quality**: Automated testing and linting
- **Documentation**: Keep architecture docs updated
- **Performance Monitoring**: Regular performance reviews
- **Security Updates**: Dependency vulnerability scanning