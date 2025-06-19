# Universal Programmable Episode Indexing System

**One Docker. One API. Infinite Selectors.**

## Quick Start

```bash
# 1. Select episodes (random 20)
python universal_processor.py selectors/random_20.py

# 2. Start API
python universal_api.py

# 3. Search
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "Churchill", "limit": 3}'
```

## Architecture

```
selectors/your_script.py → universal_processor.py → universal_api.py
     ↓                           ↓                        ↓
Select episodes            Index episodes            Serve searches
```

## Custom Selectors

Write any Python script that outputs file paths:

**`selectors/random_20.py`** - 20 random episodes
**`selectors/churchill_episodes.py`** - Episodes mentioning Churchill

Create your own:
```python
#!/usr/bin/env python3
from pathlib import Path

def select_episodes():
    podcast_dir = Path("podcast/downloads")
    # Your selection logic here
    return selected_files

if __name__ == "__main__":
    for f in select_episodes():
        print(f.absolute())
```

## Docker (Production)

```bash
# Build
docker build -f Dockerfile.universal -t transcription-search .

# Run with custom selector
docker run -p 8000:8000 \
  -v ./podcast/downloads:/app/podcast/downloads:ro \
  -e SELECTOR_SCRIPT=selectors/churchill_episodes.py \
  transcription-search

# Or use docker-compose
docker-compose -f docker-compose.universal.yml up
```

## Examples

**Random episodes:**
```bash
python universal_processor.py selectors/random_20.py
```

**Churchill episodes:**
```bash  
python universal_processor.py selectors/churchill_episodes.py
```

**Your custom selector:**
```bash
python universal_processor.py selectors/my_custom_selector.py
```

The API auto-discovers whatever index was built and serves it with episode metadata.

## Files Created

- **Selector**: `selectors/your_script.py` (your episode selection logic)
- **Processor**: `universal_processor.py` (feeds selector → indexer)  
- **API**: `universal_api.py` (auto-discovers and serves any index)
- **Docker**: `Dockerfile.universal` + `docker-compose.universal.yml`

**Total: 4 core files. Zero configuration. Maximum flexibility.** 