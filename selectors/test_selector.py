#!/usr/bin/env python3
from pathlib import Path
import sys

def select_episodes():
    """Select a few example episodes from the local data directory"""
    
    # In Docker, look for any existing transcript files in data/
    data_dir = Path("/app/data")
    if data_dir.exists():
        # Find any .txt files that might be episode transcripts
        txt_files = list(data_dir.glob("*.txt"))
        if txt_files:
            print(f"Found {len(txt_files)} text files in data/", file=sys.stderr)
            return txt_files[:5]  # Take up to 5 files
    
    # Create mock episodes for testing if no files found
    print("No existing files found, creating test episodes", file=sys.stderr)
    
    # Create some test episode files
    test_episodes = []
    for i in range(3):
        episode_file = data_dir / f"test_episode_{i+1}.txt"
        episode_content = f"""
Test Episode {i+1} Transcript

This is a sample transcript for testing the universal transcription search system.
It contains various topics and keywords for demonstration purposes.

Topics covered:
- History and historical events
- Churchill and World War II references
- Technology and innovation
- Cultural discussions

The content is generated automatically for testing the indexing and search capabilities
of the memvid system. Each episode contains unique identifiers and searchable content.

Episode duration: 45 minutes
Date: 2024-01-{i+1:02d}
Host: Test Host {i+1}
        """.strip()
        
        episode_file.write_text(episode_content)
        test_episodes.append(episode_file)
        print(f"Created test episode: {episode_file}", file=sys.stderr)
    
    return test_episodes

if __name__ == "__main__":
    selected = select_episodes()
    for f in selected:
        print(f.absolute()) 