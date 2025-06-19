# Universal Transcription Search System

🚀 **Single Docker, Single API, Infinite Selectors** - A production-ready transcription search system with programmable episode selection and zero configuration sprawl.

## 🎯 What This Is

A universal Docker system that can index any subset of podcast transcripts through **programmable selectors** and serve them via a unified search API. Write a simple Python script that outputs file paths, and the system handles the rest.

## ✨ Key Features

- **🔧 Universal Architecture**: One Docker image works with any selector script
- **📜 Programmable Selection**: Write simple Python scripts to choose episodes
- **🔍 Auto-Discovery**: API automatically discovers and serves whatever was indexed
- **⚡ Zero Configuration**: No hardcoded paths, datasets, or ports
- **🏗️ Production Ready**: Health checks, monitoring, and volume persistence
- **🔄 Infinite Extensibility**: Add new selection strategies without rebuilding

## 🚀 Quick Start

### 1. Setup Your Transcripts

```bash
# Place your podcast transcripts in the podcast directory
mkdir -p podcast/downloads
cp your_transcripts/*.txt podcast/downloads/
```

### 2. Choose Your Episodes (Selector Scripts)

**Option A: Use Built-in Selectors**
```bash
# 20 random episodes
export SELECTOR_SCRIPT=selectors/random_20.py

# Latest 50 episodes  
export SELECTOR_SCRIPT=selectors/latest_50.py

# Episodes mentioning Churchill
export SELECTOR_SCRIPT=selectors/churchill_episodes.py

# Custom keyword search
export SELECTOR_SCRIPT=selectors/keyword_search.py
export SEARCH_KEYWORD="artificial intelligence"
```

**Option B: Write Your Own Selector**
```python
# selectors/my_custom_selector.py
from pathlib import Path

def select_episodes():
    podcast_dir = Path("podcast/downloads")
    # Your custom logic here
    selected_files = []
    for file_path in podcast_dir.glob("*.txt"):
        # Add your selection criteria
        if "machine_learning" in file_path.read_text().lower():
            selected_files.append(file_path)
    return selected_files[:10]  # Top 10

if __name__ == "__main__":
    for f in select_episodes():
        print(f)
```

### 3. Run the Universal System

**Local Development:**
```bash
# Install dependencies
pip install -r requirements.txt

# Run the universal pipeline
./run_universal.sh

# Or specify a custom selector
SELECTOR_SCRIPT=selectors/my_custom_selector.py ./run_universal.sh
```

**Production Docker:**
```bash
# Build and run
docker-compose -f docker-compose.production.yml up -d

# Or with custom selector
SELECTOR_SCRIPT=selectors/my_custom_selector.py \
docker-compose -f docker-compose.production.yml up -d
```

### 4. Search Your Episodes

```bash
# Health check
curl http://localhost:8000/health

# Search for content
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "Churchill", "limit": 3}'

# Get system info
curl http://localhost:8000/info
```

## 📁 Project Structure

```
├── selectors/                    # Episode selection scripts
│   ├── random_20.py             # Select 20 random episodes
│   ├── latest_50.py             # Latest 50 by modification time
│   ├── churchill_episodes.py    # Episodes mentioning Churchill
│   ├── keyword_search.py        # Configurable keyword search
│   └── your_custom_selector.py  # Your custom selection logic
├── universal_processor.py       # Universal processing pipeline
├── universal_api.py             # Auto-discovering search API
├── run_universal.sh             # Universal runner script
├── Dockerfile.production        # Production Docker build
├── docker-compose.production.yml # Production deployment
├── core/                        # Core processing modules
├── podcast/downloads/           # Your transcript files
└── data/                        # Generated indexes and mappings
```

## 🛠️ How It Works

### 1. Selector Pattern
Every selector script follows this simple pattern:
```python
def select_episodes():
    # Your selection logic here
    return list_of_file_paths

if __name__ == "__main__":
    for file_path in select_episodes():
        print(file_path)
```

### 2. Universal Processing
```bash
# The system automatically:
# 1. Runs your selector script
# 2. Processes selected transcripts into video format
# 3. Builds semantic search index
# 4. Creates episode metadata mapping
# 5. Starts API server with auto-discovered data
```

### 3. Auto-Discovery API
The API automatically discovers and serves whatever index was built:
- Finds the latest video file in `data/`
- Loads corresponding index and mapping files
- Provides search, health, and info endpoints

## 🔧 Built-in Selectors

### Random Episodes
```bash
SELECTOR_SCRIPT=selectors/random_20.py
# Selects 20 random episodes from your collection
```

### Latest Episodes
```bash
SELECTOR_SCRIPT=selectors/latest_50.py
# Selects the 50 most recently modified episodes
```

### Keyword-Based Selection
```bash
SELECTOR_SCRIPT=selectors/keyword_search.py
SEARCH_KEYWORD="machine learning"
# Selects episodes containing the specified keyword
```

### Topic-Specific Selection  
```bash
SELECTOR_SCRIPT=selectors/churchill_episodes.py
# Selects episodes that mention Churchill
```

## 🐳 Production Deployment

### Environment Configuration
```yaml
# docker-compose.production.yml
environment:
  - SELECTOR_SCRIPT=selectors/your_selector.py
  - SEARCH_KEYWORD=your_keyword        # For keyword_search.py
  - PYTHONUNBUFFERED=1
  - MAX_MEMORY_MB=1024
  - OMP_NUM_THREADS=2
```

### Volume Mounts
```yaml
volumes:
  # Mount your transcript collection (read-only)
  - ./podcast/downloads:/app/podcast/downloads:ro
  # Optionally mount custom selectors
  - ./selectors:/app/selectors:ro
  # Persist generated data
  - transcription_data:/app/data
```

### Health Monitoring
```bash
# Built-in health checks every 30 seconds
curl http://localhost:8000/health

# Response format:
{
  "status": "healthy",
  "search_engine": true,
  "episodes_loaded": 20,
  "index_file": "data/selected_episodes.mp4"
}
```

## 🔍 API Reference

### Search Endpoint
```bash
POST /search
Content-Type: application/json

{
  "query": "artificial intelligence",
  "limit": 5
}

# Response:
{
  "results": [
    {
      "text": "transcript text snippet...",
      "episode": {
        "title": "Episode Title",
        "file_path": "/path/to/episode.txt",
        "frame_number": 42
      }
    }
  ],
  "total_results": 1,
  "query": "artificial intelligence"
}
```

### System Info
```bash
GET /info

# Response:
{
  "total_episodes": 20,
  "total_chunks": 2075,
  "index_file": "data/selected_episodes.mp4",
  "selector_used": "selectors/random_20.py"
}
```

### Health Check
```bash
GET /health

# Response:
{
  "status": "healthy",
  "search_engine": true,
  "episodes_loaded": 20
}
```

## ✍️ Writing Custom Selectors

### Basic Template
```python
#!/usr/bin/env python3
from pathlib import Path
import os

def select_episodes():
    """Your custom episode selection logic"""
    # Access environment variables
    keyword = os.getenv('SEARCH_KEYWORD', 'default')
    
    # Look for transcripts
    podcast_dir = Path("/app/podcast/downloads") if Path("/app").exists() else Path("podcast/downloads")
    
    selected_files = []
    for file_path in podcast_dir.glob("*.txt"):
        # Add your selection criteria here
        if meets_your_criteria(file_path):
            selected_files.append(file_path)
    
    return selected_files

def meets_your_criteria(file_path):
    """Implement your selection logic"""
    try:
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        # Your logic here
        return True  # or False
    except:
        return False

if __name__ == "__main__":
    selected = select_episodes()
    print(f"Selected {len(selected)} episodes", file=sys.stderr)
    for f in selected:
        print(f)
```

### Advanced Examples

**Date-Based Selection:**
```python
def select_episodes():
    from datetime import datetime, timedelta
    cutoff = datetime.now() - timedelta(days=30)
    
    files_with_dates = []
    for file_path in podcast_dir.glob("*.txt"):
        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
        if mtime > cutoff:
            files_with_dates.append((mtime, file_path))
    
    # Return newest first
    files_with_dates.sort(reverse=True)
    return [f for _, f in files_with_dates[:25]]
```

**Content Analysis Selection:**
```python
def select_episodes():
    import re
    
    # Select episodes with high technical content
    technical_patterns = [
        r'\b(algorithm|database|API|machine learning)\b',
        r'\b(Python|JavaScript|Docker|Kubernetes)\b',
    ]
    
    selected = []
    for file_path in podcast_dir.glob("*.txt"):
        content = file_path.read_text(errors='ignore').lower()
        
        tech_score = sum(
            len(re.findall(pattern, content, re.IGNORECASE))
            for pattern in technical_patterns
        )
        
        if tech_score >= 5:  # At least 5 technical mentions
            selected.append(file_path)
    
    return selected[:30]
```

## 🔧 Configuration

### Environment Variables
```bash
# Core settings
SELECTOR_SCRIPT=selectors/your_selector.py
SEARCH_KEYWORD=your_keyword

# Performance tuning
MAX_MEMORY_MB=1024
OMP_NUM_THREADS=2
PYTHONUNBUFFERED=1

# Docker settings
COMPOSE_PROJECT_NAME=transcription-search
```

### Resource Requirements
- **Memory**: 1-2GB RAM for processing, 512MB for serving
- **CPU**: 2+ cores recommended for processing
- **Storage**: ~5MB per episode for indexes
- **Network**: Port 8000 for API access

## 🚀 Advanced Usage

### Multiple Deployments
```bash
# Deploy different episode collections on different ports
SELECTOR_SCRIPT=selectors/tech_episodes.py docker-compose -p tech up -d
SELECTOR_SCRIPT=selectors/history_episodes.py docker-compose -p history up -d
```

### Custom Processing Parameters
```python
# In your selector script, return metadata too:
def select_episodes():
    return {
        "files": [list_of_files],
        "processing_params": {
            "chunk_size": 1000,
            "overlap": 200,
            "fps": 10
        }
    }
```

### Monitoring & Logs
```bash
# View processing logs
docker logs transcription-search-universal

# Monitor resource usage
docker stats transcription-search-universal

# Check health status
watch -n 5 "curl -s http://localhost:8000/health | jq"
```

## 🐛 Troubleshooting

### Common Issues

**Port Already in Use:**
```bash
# Change port in docker-compose.yml
ports:
  - "8001:8000"  # Use port 8001 instead
```

**Selector Script Not Found:**
```bash
# Ensure script exists and is executable
ls -la selectors/your_selector.py
chmod +x selectors/your_selector.py
```

**No Episodes Selected:**
```bash
# Test your selector directly
python selectors/your_selector.py
```

**Memory Issues:**
```bash
# Reduce memory usage in docker-compose.yml
environment:
  - MAX_MEMORY_MB=512
  - OMP_NUM_THREADS=1
```

### Debug Mode
```bash
# Run with debug logging
docker-compose -f docker-compose.production.yml up
# Watch logs in real-time
```

## 📊 Performance Tuning

### Optimization Tips

1. **Selector Efficiency**: Make selectors fast by avoiding reading all files
2. **Memory Management**: Tune `MAX_MEMORY_MB` based on available resources  
3. **CPU Usage**: Adjust `OMP_NUM_THREADS` for your system
4. **Index Size**: More episodes = larger indexes but better search quality

### Scaling Considerations

- **<100 episodes**: Single container, 512MB RAM
- **100-500 episodes**: 1GB RAM, 2 CPU cores
- **500+ episodes**: 2GB RAM, 4 CPU cores, consider sharding

## 🤝 Contributing

### Adding New Selectors
1. Create your selector in `selectors/`
2. Follow the standard interface pattern
3. Add documentation in the selector file
4. Test with various episode collections

### Extending the API
1. Modify `universal_api.py`
2. Add new endpoints following existing patterns
3. Update health checks if needed
4. Maintain backward compatibility

## 📚 Examples & Use Cases

### Research & Analysis
```bash
# Index episodes from specific time periods
SELECTOR_SCRIPT=selectors/year_2023.py

# Focus on specific topics
SELECTOR_SCRIPT=selectors/ai_episodes.py
SEARCH_KEYWORD="artificial intelligence"
```

### Content Curation
```bash
# Best episodes (by length/engagement metrics)
SELECTOR_SCRIPT=selectors/top_episodes.py

# Guest-specific episodes
SELECTOR_SCRIPT=selectors/guest_episodes.py
GUEST_NAME="Tim Ferriss"
```

### Quality Control
```bash
# Episodes needing review
SELECTOR_SCRIPT=selectors/quality_check.py

# Duplicate detection
SELECTOR_SCRIPT=selectors/find_duplicates.py
```

## 🎉 Success! 

Your universal transcription search system is now running! 

- ✅ **Episodes processed and indexed**
- ✅ **Search API serving on port 8000**  
- ✅ **Health monitoring active**
- ✅ **Ready for production use**

Start searching your episodes and building custom selectors for your specific needs!

---

*Built with ❤️ using the Universal Selector Pattern - Single Docker, Single API, Infinite Possibilities*