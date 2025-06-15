# AI-Powered Podcast Search Repository - Product Requirements Document

## Overview

An AI-powered podcast search system that uses novel video encoding to store transcript chunks as compressed MP4 files, enabling sub-second semantic search across 700+ podcast episodes. The core innovation treats video frames as a compressed database for text data, targeting 100:1+ compression ratios while maintaining fast retrieval through hybrid semantic and keyword search.

**Current Implementation Status**: 25% Complete (Foundation Stage)
- ✅ Infrastructure, dependencies, and memvid integration working
- 🟡 Core modules stubbed with basic structure  
- ❌ Processing pipeline, search engine, and interfaces not implemented

## Core Features

### 1. Video-as-Database Storage
**What it does**: Encodes text transcript chunks as compressed MP4 video frames using the memvid library
**Why it's important**: Achieves massive compression ratios (target 100:1+) while maintaining fast random access to text data
**Implementation status**: ✅ Memvid integration working - successfully encoding/decoding 4 chunks to 175KB MP4 file
**How it works**: 
- Text chunks converted to QR codes embedded in video frames
- MP4 container provides compression and efficient storage
- Direct decode access without full video processing

### 2. Conversation-Aware Transcript Processing  
**What it does**: Intelligently chunks transcripts while preserving speaker context and dialogue flow
**Why it's important**: Maintains conversation coherence for better search relevance and context understanding
**Implementation status**: ❌ Not implemented - exists only as YAML configuration
**How it works**:
- Speaker turn detection and preservation
- Context-aware chunking boundaries
- Quality control filtering for low-value chunks

### 3. Hybrid Semantic + Keyword Search
**What it does**: Combines FAISS vector similarity search with traditional keyword matching for optimal results
**Why it's important**: Balances semantic understanding with exact term matching for comprehensive search coverage
**Implementation status**: ❌ Not implemented - FAISS dependency installed but no search logic
**How it works**:
- Sentence-transformers embeddings for semantic similarity
- FAISS indexing for fast vector search (<1 second across 700+ episodes)
- Weighted result fusion with deduplication

### 4. Incremental Processing Pipeline
**What it does**: Processes new episodes without rebuilding entire search index
**Why it's important**: Enables scalable addition of content without performance degradation
**Implementation status**: ❌ Not implemented - no processing scripts exist
**How it works**:
- Streaming processing for memory efficiency
- Embedding caching to avoid recomputation
- Delta indexing for new content integration

### 5. Fast API and Web Interface
**What it does**: Provides REST API endpoints and browser-based search interface
**Why it's important**: Enables integration with other systems and user-friendly search experience
**Implementation status**: ❌ Not implemented - FastAPI dependency installed, no routes defined
**How it works**:
- RESTful search endpoints with multiple query modes
- Advanced filtering (episodes, dates, speakers)
- Result highlighting and context expansion

## User Experience

### User Personas
1. **Podcast Researchers**: Academics and journalists searching for specific topics across multiple shows
2. **Content Creators**: Podcasters looking for reference material and avoiding topic duplication  
3. **Knowledge Workers**: Professionals seeking insights and quotes from business/tech podcasts
4. **General Listeners**: Fans wanting to rediscover specific conversations or moments

### Key User Flows
1. **Quick Search**: Enter query → get ranked results → click for context → view full episode
2. **Advanced Search**: Filter by podcast, date range, speakers → refined results → export/save
3. **Topic Discovery**: Browse by themes → explore related episodes → build research collections
4. **Context Expansion**: Find specific moment → see surrounding conversation → navigate to full transcript

### UI/UX Considerations
- **Search Speed**: Sub-second response time across full 700+ episode corpus
- **Result Relevance**: Clear highlighting of matched terms and semantic similarity scores
- **Context Preservation**: Show conversation flow around search results
- **Cross-Episode Discovery**: Surface related content from different shows

## Technical Architecture

### System Components

#### Current Implementation (Working)
- **Virtual Environment**: Python 3.13.5 with 107 dependencies successfully installed
- **Memvid Integration**: Text-to-video encoding/decoding functional (memvid==0.1.3)
- **ML Dependencies**: PyTorch 2.7.1, sentence-transformers 4.1.0, FAISS 1.11.0 ready
- **Web Framework**: FastAPI 0.115.12 installed and ready for endpoint development
- **Configuration System**: Comprehensive YAML configs for embeddings, chunking, and search
- **Docker Infrastructure**: Complete Dockerfile and docker-compose.yml for deployment

#### Planned Components (Not Implemented)
- **Transcript Processor**: Conversation-aware chunking with speaker detection
- **Embedding Generator**: Sentence-transformer integration with caching
- **Search Engine**: FAISS indexing with hybrid search algorithms
- **Storage Manager**: Video library organization and metadata management
- **API Layer**: RESTful endpoints for search, episodes, and health monitoring
- **Web Interface**: Browser-based search with advanced filtering

### Data Models

#### Episode Structure
```python
Episode {
    id: str                    # Unique episode identifier
    title: str                 # Episode title
    podcast_name: str          # Source podcast name
    date: datetime             # Publication date
    duration: int              # Episode length in seconds
    speakers: List[str]        # Identified speakers
    video_file: str            # Path to MP4 storage file
    chunk_count: int           # Number of text chunks
    metadata: dict             # Additional episode information
}
```

#### Chunk Structure  
```python
Chunk {
    id: str                    # Unique chunk identifier  
    episode_id: str            # Parent episode reference
    start_time: float          # Timestamp in episode
    end_time: float            # End timestamp
    speaker: str               # Primary speaker
    text: str                  # Transcript content
    embedding: List[float]     # Semantic vector (384-dim)
    quality_score: float       # Content quality metric
}
```

### APIs and Integrations

#### Core API Endpoints (Planned)
- `POST /api/v1/search`: Main search with semantic/keyword/hybrid modes
- `GET /api/v1/episodes`: List episodes with metadata and filtering
- `GET /api/v1/episodes/{id}`: Episode details and full transcript access
- `GET /api/v1/health`: System status and performance metrics

#### External Dependencies
- **sentence-transformers**: Pre-trained models for text embeddings
- **FAISS**: Facebook's similarity search library for vector indexing
- **memvid**: Video encoding library for text-to-video compression
- **OpenCV**: Video processing and frame manipulation
- **FastAPI**: Modern async web framework for API development

### Infrastructure Requirements

#### Development Environment
- **Python**: 3.13.5 with virtual environment (✅ Working)
- **Storage**: ~50GB for 700 episodes (target 5MB per episode average)
- **Memory**: 16GB RAM recommended for embedding generation and search
- **Docker**: Containerized deployment ready

#### Production Deployment  
- **Container**: Docker with multi-stage builds for optimization
- **Storage**: Network-attached storage for video libraries and indexes
- **Compute**: GPU optional for embedding generation (CPU fallback available)
- **Scaling**: Horizontal scaling for API layer, vertical for search performance

## Development Roadmap

### Phase 1: MVP Core Engine (Weeks 1-4)
**Goal**: Working search system with basic functionality

**Components to Build**:
- **Embedding Generator** (`core/embeddings.py`): Implement sentence-transformers integration
  - Load and cache pre-trained models (all-MiniLM-L6-v2)
  - Batch processing for multiple chunks
  - Embedding vector storage and retrieval
  
- **Search Engine** (`core/search_engine.py`): Build FAISS-based search
  - Vector indexing and similarity search
  - Keyword search implementation  
  - Result ranking and fusion algorithms
  
- **Storage Manager** (`core/storage.py`): File system organization
  - Video library management
  - Metadata indexing and caching
  - Chunk-to-video mapping

- **Basic API** (`api/search_api.py`): Essential endpoints
  - Search endpoint with query processing
  - Health check and status monitoring
  - Error handling and response formatting

**Deliverables**:
- Search 10-20 test episodes successfully
- API responds to queries in <2 seconds
- Basic web interface for testing

### Phase 2: Transcript Processing Pipeline (Weeks 5-8)  
**Goal**: Automated ingestion and processing of new transcripts

**Components to Build**:
- **Transcript Processor** (`podcast/transcript_processor.py`): Parse and chunk transcripts
  - Support multiple formats (TXT, SRT, JSON)
  - Conversation-aware chunking with speaker detection
  - Quality control and filtering

- **Batch Processing** (`scripts/process_all_episodes.py`): Mass processing capability
  - Streaming processing for memory efficiency
  - Progress tracking and error recovery
  - Incremental updates and delta processing

- **Data Validation** (`tests/test_processing.py`): Quality assurance
  - Chunk quality validation
  - Embedding consistency checks
  - Search result accuracy testing

**Deliverables**:
- Process 100+ episodes automatically
- Validated chunk quality and search accuracy
- Monitoring and alerting for processing failures

### Phase 3: Advanced Search Features (Weeks 9-12)
**Goal**: Production-ready search with advanced capabilities

**Components to Build**:
- **Hybrid Search Enhancement**: Optimize ranking algorithms
  - Advanced result fusion strategies
  - Personalization and user feedback integration
  - Context expansion and related content discovery

- **Advanced Filtering**: Complex query capabilities
  - Date range, podcast, and speaker filtering
  - Topic clustering and thematic search
  - Cross-episode topic discovery

- **Performance Optimization**: Sub-second search at scale
  - FAISS index optimization for 700+ episodes
  - Caching strategies for frequent queries
  - Concurrent search handling

**Deliverables**:
- Search across full 700+ episode corpus in <1 second
- Advanced filtering and topic discovery working
- Performance benchmarks meeting production targets

### Phase 4: Production Interface (Weeks 13-16)
**Goal**: User-facing web application and API optimization

**Components to Build**:
- **Web Interface** (`web/app.py`): Full-featured search application
  - Advanced search forms with filtering
  - Result highlighting and context windows
  - Export functionality (CSV, JSON)

- **API Enhancement**: Production-grade endpoints
  - Authentication and rate limiting
  - Comprehensive error handling
  - API documentation and testing

- **Deployment Pipeline**: Production deployment
  - Docker optimization and security
  - Monitoring and logging integration
  - Backup and disaster recovery

**Deliverables**:
- Production-ready web application
- Comprehensive API with documentation
- Deployed system handling concurrent users

## Logical Dependency Chain

### Foundation Layer (Must Build First)
1. **Core Infrastructure** (✅ Complete)
   - Virtual environment with dependencies
   - Memvid integration and testing
   - Configuration system and Docker setup

2. **Storage and Encoding** (Week 1)
   - Complete `core/storage.py` implementation
   - Optimize memvid compression ratios
   - Video library organization system

3. **Embedding System** (Week 1-2)  
   - Implement `core/embeddings.py` with sentence-transformers
   - Batch processing and caching mechanisms
   - Vector storage and retrieval optimization

### Search Engine Layer (Builds on Foundation)
4. **Basic Search** (Week 2-3)
   - FAISS indexing in `core/search_engine.py`
   - Vector similarity search implementation
   - Basic result ranking and retrieval

5. **Hybrid Search** (Week 3-4)
   - Keyword search integration
   - Result fusion algorithms
   - Query preprocessing and optimization

### Application Layer (Builds on Search)
6. **API Development** (Week 4-5)
   - Core search endpoints in `api/search_api.py`
   - Request/response handling
   - Error management and validation

7. **Processing Pipeline** (Week 5-6)
   - Transcript parsing and chunking
   - Batch processing scripts
   - Quality control and validation

### User Interface Layer (Final Integration)
8. **Web Interface** (Week 7-8)
   - Search interface and result display
   - Advanced filtering and export features
   - User experience optimization

### Getting to Usable Frontend Quickly

**Week 2 Milestone**: Basic searchable prototype
- Implement embeddings with cached results for 10 test episodes
- Build minimal search API endpoint
- Create simple HTML form for query input and result display
- Focus on proving core concept works end-to-end

**Quick Win Strategy**:
- Use pre-computed embeddings for initial testing
- Simple JavaScript frontend calling FastAPI backend
- Hardcode 10-20 test episodes to validate entire pipeline
- Prioritize working demonstration over optimization

### Atomic Feature Scoping

Each feature is designed to be:
- **Self-contained**: Can be built and tested independently
- **Incremental**: Adds value to existing functionality
- **Extensible**: Can be enhanced in future iterations

**Example - Embedding System**:
- **Atomic Version**: Single-threaded processing with basic sentence-transformers
- **Enhancement 1**: Batch processing for performance
- **Enhancement 2**: GPU acceleration and model optimization
- **Enhancement 3**: Custom fine-tuning for podcast content

## Risks and Mitigations

### Technical Challenges

**1. Compression Ratio Performance**
- **Risk**: Memvid may not achieve target 100:1 compression ratios
- **Current Status**: Test shows 0.00:1 ratio (175KB video for 213 bytes text)
- **Mitigation**: Optimize chunk sizes, explore alternative encoding parameters, benchmark against traditional storage

**2. Search Performance at Scale**
- **Risk**: FAISS indexing may not meet <1 second search requirement for 700+ episodes
- **Mitigation**: Index optimization, caching strategies, incremental loading, hardware scaling

**3. Memory Management**
- **Risk**: Loading embeddings and videos may exceed available RAM
- **Mitigation**: Streaming processing, embedding compression, lazy loading, distributed processing

### MVP Scope Definition

**Core MVP Requirements**:
1. **Working Search**: Query 50+ episodes with semantic and keyword search
2. **Video Storage**: Demonstrate compression and retrieval from MP4 files
3. **Basic API**: REST endpoints for search and episode listing
4. **Simple Interface**: Web form for search with result display

**MVP Success Criteria**:
- End-to-end pipeline from transcript to searchable results
- Search response time <5 seconds (optimized later)
- Compression ratio >10:1 (path to 100:1 demonstrated)
- Concurrent handling of 5+ users

**Out of MVP Scope**:
- Advanced filtering and topic discovery
- Production-grade authentication and security
- Extensive podcast format support
- Real-time processing and notifications

### Resource Constraints

**Development Resources**:
- **Primary Developer**: Full-stack development across ML, backend, and frontend
- **Infrastructure**: Local development with cloud deployment option
- **Data**: 700+ episodes requiring processing and storage management

**Technical Debt Management**:
- **Code Quality**: Comprehensive testing suite and type hints
- **Documentation**: API documentation and architectural decision records
- **Performance**: Profiling and optimization checkpoints at each phase

## Appendix

### Research Findings

**Memvid Integration Success**:
- Successfully encoded 4 text chunks to MP4 format
- Decode/encode cycle working without data loss
- Compression ratio needs optimization but core concept proven

**Architecture Validation**:
- All major dependencies (FAISS, sentence-transformers, FastAPI) compatible
- Configuration system comprehensive and flexible
- Docker infrastructure ready for deployment

**Performance Baseline**:
- sentence-transformers model (all-MiniLM-L6-v2) chosen for balance of speed and accuracy
- FAISS CPU version selected for compatibility (GPU optional)
- Target metrics: <1 second search, <5MB per episode, >95% relevance

### Technical Specifications

**Environment Setup**:
```bash
# Activate virtual environment (required for all development)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Start development API
python -m uvicorn api.search_api:app --reload --port 8000

# Process episodes
python scripts/process_all_episodes.py
```

**Key Dependencies**:
- **Python**: 3.13.5
- **memvid**: 0.1.3 (video encoding)
- **sentence-transformers**: 4.1.0 (embeddings)
- **faiss-cpu**: 1.11.0 (vector search)
- **fastapi**: 0.115.12 (API framework)
- **torch**: 2.7.1 (ML backend)

**Configuration Files**:
- `config/embeddings.yaml`: Model selection and caching
- `config/chunking.yaml`: Transcript processing parameters
- `config/search.yaml`: Search and ranking configuration
- `.env`: Runtime environment variables and secrets

**Current File Structure**:
```
├── core/                 # Foundation modules (25% complete)
├── podcast/              # Transcript processing (0% complete)  
├── api/                  # FastAPI endpoints (0% complete)
├── web/                  # Web interface (0% complete)
├── config/               # YAML configurations (100% complete)
├── data/                 # Data storage (structure ready)
├── scripts/              # Processing pipelines (0% complete)
├── tests/                # Test framework (0% complete)
├── requirements.txt      # Dependencies (100% complete)
├── Dockerfile           # Container setup (100% complete)
└── CLAUDE.md            # Development guidance (100% complete)
```

The project has excellent architectural foundation and proven core technology (memvid) but requires significant development effort to implement the processing pipeline, search engine, and user interfaces described in this PRD.