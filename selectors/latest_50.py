#!/usr/bin/env python3
from pathlib import Path
import os

def select_episodes():
    """Select the latest 50 episodes by modification time"""
    podcast_dir = Path("/app/podcast/downloads") if Path("/app").exists() else Path("podcast/downloads")
    
    # Get all files with their modification times
    files_with_times = []
    for file_path in podcast_dir.glob("*.txt"):
        try:
            mtime = os.path.getmtime(file_path)
            files_with_times.append((mtime, file_path))
        except:
            continue
    
    # Sort by modification time (newest first) and take top 50
    files_with_times.sort(reverse=True)
    latest_files = [file_path for _, file_path in files_with_times[:50]]
    
    return latest_files

if __name__ == "__main__":
    selected = select_episodes()
    for f in selected:
        print(f.absolute()) 