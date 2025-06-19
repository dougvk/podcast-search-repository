#!/usr/bin/env python3
from pathlib import Path

def select_episodes():
    """Select episodes that mention Churchill"""
    podcast_dir = Path("/app/podcast/downloads") if Path("/app").exists() else Path("podcast/downloads")
    churchill_files = []
    
    for file_path in podcast_dir.glob("*.txt"):
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore').lower()
            if 'churchill' in content:
                churchill_files.append(file_path)
        except:
            continue
    
    return churchill_files[:50]  # Max 50 episodes

if __name__ == "__main__":
    selected = select_episodes()
    for f in selected:
        print(f.absolute()) 