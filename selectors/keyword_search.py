#!/usr/bin/env python3
from pathlib import Path
import os
import sys

def select_episodes():
    """Select episodes containing specific keywords"""
    # Get keyword from environment or command line
    keyword = os.getenv('SEARCH_KEYWORD', 'Churchill')
    if len(sys.argv) > 1:
        keyword = sys.argv[1]
    
    print(f"Searching for episodes containing: {keyword}", file=sys.stderr)
    
    podcast_dir = Path("/app/podcast/downloads") if Path("/app").exists() else Path("podcast/downloads")
    matching_files = []
    
    for file_path in podcast_dir.glob("*.txt"):
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore').lower()
            if keyword.lower() in content:
                matching_files.append(file_path)
        except:
            continue
    
    print(f"Found {len(matching_files)} episodes containing '{keyword}'", file=sys.stderr)
    return matching_files[:100]  # Max 100 episodes

if __name__ == "__main__":
    selected = select_episodes()
    for f in selected:
        print(f.absolute()) 