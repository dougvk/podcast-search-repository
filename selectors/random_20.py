#!/usr/bin/env python3
import random
from pathlib import Path

def select_episodes():
    """Select 20 random episodes from all available transcripts"""
    podcast_dir = Path("/app/podcast/downloads") if Path("/app").exists() else Path("podcast/downloads")
    all_files = list(podcast_dir.glob("*.txt"))
    return random.sample(all_files, min(20, len(all_files)))

if __name__ == "__main__":
    selected = select_episodes()
    for f in selected:
        print(f.absolute()) 