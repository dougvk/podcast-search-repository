# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI-powered podcast search repository that uses novel video encoding to store transcript chunks as compressed MP4 files, enabling sub-second semantic search across 700+ podcast episodes. The core innovation is treating video frames as a compressed database for text data, achieving 100:1+ compression ratios while maintaining fast retrieval.

## Architecture Overview

### Core Data Flow
1. **Transcript Processing** (`podcast/`): Raw transcripts → conversation-aware chunks → metadata extraction
2. **Video Encoding** (`core/video_encoder.py`): Text chunks → MP4 video frames (memvid-based)
3. **Embedding Generation** (`core/embeddings.py`): Chunks → semantic vectors (sentence-transformers)
4. **Search Indexing** (`core/search_engine.py`): Vectors → FAISS indices → hybrid search
5. **API/Web Layer** (`api/`, `web/`): User queries → search results → context retrieval

### Key Components Integration
- **Video-as-Database**: Text chunks encoded as video frames using OpenCV, stored as MP4 files
- **Hybrid Search**: Combines FAISS semantic similarity with keyword search for optimal results
- **Conversation-Aware Processing**: Preserves speaker turns and context boundaries during chunking
- **Incremental Processing**: New episodes can be added without rebuilding entire index

## Development Commands

### Environment Setup
```bash
# Local development
cp .env.example .env
pip install -r requirements.txt

# Docker development
docker-compose up -d

# Production deployment
docker-compose -f docker-compose.prod.yml up -d
```

### Running Services
```bash
# API server
python -m uvicorn api.search_api:app --reload --port 8000

# Web interface
python web/app.py --port 8080

# Full processing pipeline
python scripts/process_all_episodes.py
```

### Testing and Quality
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_transcript_processor.py -v

# Run tests with coverage
pytest tests/ --cov=core --cov=podcast --cov=api

# Code quality checks
black .
flake8 .
mypy .
```

### Data Processing
```bash
# Process new transcripts
python scripts/process_all_episodes.py --input data/raw_transcripts/

# Rebuild search index
python scripts/rebuild_index.py

# Benchmark search performance
python scripts/benchmark_search.py
```

## Configuration Architecture

### Configuration Hierarchy
1. **Environment Variables** (`.env`): Runtime settings, secrets, deployment config
2. **YAML Configs** (`config/`): Feature behavior, model parameters, processing settings
3. **Code Defaults**: Fallback values in Python modules

### Critical Configuration Files
- `config/embeddings.yaml`: Model selection, caching, conversation-aware settings
- `config/chunking.yaml`: Text processing, speaker handling, quality control
- `config/search.yaml`: FAISS parameters, ranking, result formatting

## Implementation Status

**Current Stage**: Foundation Setup (Stage 1) - 15% Complete
**Next Priority**: Memvid integration for video encoding/decoding

### Development Stages
1. **Foundation** (Days 1-7): Environment, core modules, memvid integration
2. **Processing** (Days 8-14): Transcript parsing, chunking, batch processing
3. **Search** (Days 15-21): Embeddings, FAISS indexing, video storage integration
4. **Interface** (Days 22-28): API endpoints, web interface, deployment

### Progress Tracking
Check `tracking/PROGRESS.md` for current status and `tracking/stages/` for detailed stage breakdowns.

## Data Processing Pipeline

### Transcript → Chunks → Video → Index Flow
1. **Input**: Raw transcript files in `data/raw_transcripts/` (txt, srt, json formats)
2. **Chunking**: Conversation-aware segmentation preserving speaker context
3. **Video Encoding**: Chunks compressed into MP4 files in `data/video_libraries/`
4. **Embedding**: Semantic vectors generated and cached
5. **Indexing**: FAISS indices built in `data/indexes/`
6. **Search**: Sub-second retrieval via hybrid semantic + keyword search

### Key Processing Considerations
- **Conversation Context**: Chunking preserves speaker turns and dialogue flow
- **Memory Management**: Streaming processing for large datasets (700+ episodes)
- **Quality Control**: Filtering low-quality chunks, handling transcript variations
- **Incremental Updates**: Adding new episodes without full reprocessing

## Search Architecture

### Hybrid Search Strategy
- **Semantic Search**: sentence-transformers embeddings + FAISS similarity
- **Keyword Search**: Traditional text matching with fuzzy support
- **Result Fusion**: Weighted combination with deduplication
- **Context Expansion**: Retrieve surrounding chunks for full conversation context

### Performance Targets
- **Search Speed**: <1 second across 700+ episodes
- **Storage Efficiency**: <5MB per episode average
- **Search Accuracy**: >95% relevance for semantic queries
- **Concurrent Users**: 20+ simultaneous searches

## API Design

### Core Endpoints
- `POST /api/v1/search`: Main search with semantic/keyword/hybrid modes
- `GET /api/v1/episodes`: List all episodes with metadata
- `GET /api/v1/episodes/{id}`: Episode details and full transcript
- `GET /api/v1/health`: System status and performance metrics

### Web Interface Features
- Advanced search with filters (episodes, dates, speakers)
- Result highlighting and context windows
- Export functionality (CSV, JSON)
- Cross-episode discovery and topic clustering

## Development Notes

### Module Dependencies
- `core/` modules are foundational and imported by `podcast/` and `api/`
- `podcast/` processing pipeline feeds into `core/` storage and search
- `api/` and `web/` are presentation layers consuming `core/` functionality
- Configuration in `config/` drives behavior across all modules

### Testing Strategy
- Unit tests for individual components in each module
- Integration tests for full processing pipeline
- Performance tests with sample datasets
- API tests for all endpoints and edge cases

### Memory and Performance
- Stream processing for large transcript batches
- Embedding caching to avoid recomputation
- FAISS index optimization for search speed
- Video compression tuning for storage efficiency