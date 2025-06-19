#!/usr/bin/env python3
import sys
import subprocess
from pathlib import Path
from core.batch_processor import MemvidBatchProcessor, ProcessingConfig
import json
import re

def extract_episode_info(filename):
    """Extract episode metadata from filename"""
    name = Path(filename).stem
    episode_match = re.search(r'(\d{3,})', name)
    episode_num = episode_match.group(1) if episode_match else None
    return {
        "filename": name + ".txt",
        "title": name.replace('_', ' ').title(),
        "episode_number": episode_num,
        "speaker": "Podcast Host",
        "timestamp": None
    }

def main():
    selector_script = sys.argv[1] if len(sys.argv) > 1 else "selectors/random_20.py"
    
    # Run selector to get episode list
    result = subprocess.run([sys.executable, selector_script], capture_output=True, text=True)
    episode_files = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
    
    print(f"Selected {len(episode_files)} episodes")
    
    # Create config
    config = ProcessingConfig(max_memory_mb=512, chunk_size=1024)
    state_dir = Path("data/state")
    
    # Process episodes
    processor = MemvidBatchProcessor(config, state_dir)
    
    # Create temp dir with selected files
    temp_dir = Path("data/temp_selected")
    temp_dir.mkdir(exist_ok=True)
    
    # Copy selected files to temp dir
    for i, file_path in enumerate(episode_files):
        dest = temp_dir / f"{i:03d}_{Path(file_path).name}"
        dest.write_text(Path(file_path).read_text())
    
    # Process the temp directory
    output_dir = Path("data")
    result = processor.process_directory(temp_dir, output_dir, "selected_episodes")
    
    # Create episode mapping
    episode_mapping = {}
    for i, file_path in enumerate(episode_files):
        episode_info = extract_episode_info(file_path)
        episode_mapping[str(i)] = episode_info
    
    with open("data/episode_mapping_selected.json", "w") as f:
        json.dump(episode_mapping, f, indent=2)
    
    print(f"✅ Processed {len(episode_files)} episodes")
    print("✅ Created episode mapping")

if __name__ == "__main__":
    main() 