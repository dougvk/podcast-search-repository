#!/usr/bin/env python3

from pathlib import Path
from core.batch_processor import MemvidBatchProcessor, ProcessingConfig
import os
import json
import re
import shutil
import tempfile

def extract_episode_info(filename):
    """Extract episode metadata from filename"""
    name = Path(filename).stem
    # Extract episode number if present (e.g., "001", "Episode_123")
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
    # Create config
    config = ProcessingConfig(max_memory_mb=512, batch_size=10)
    state_dir = Path('data/batch_state')

    # Create processor
    processor = MemvidBatchProcessor(config, state_dir)

    # Process first 10 files
    input_dir = Path('podcast/downloads')
    output_dir = Path('data')

    # Get first 10 files
    txt_files = list(input_dir.glob("*.txt"))
    print(f'Found {len(txt_files)} text files')
    
    if len(txt_files) == 0:
        print("No .txt files found in podcast/downloads")
        return
    
    # Limit to first 10
    files_to_process = txt_files[:10]
    print(f'Processing first {len(files_to_process)} files for episode metadata tracking:')
    
    # **NEW: Track episode mapping**
    chunk_to_episode = {}
    current_chunk_index = 0
    
    # Create temporary directory with episode tracking
    temp_dir = Path('temp_podcasts')
    temp_dir.mkdir(exist_ok=True)
    
    try:
        for file_path in files_to_process:
            print(f'  - {file_path.name}')
            
            # Read and track chunks for this episode
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Estimate chunks (rough calculation)
            estimated_chunks = len(content) // 500  # Rough chunk size estimate
            episode_info = extract_episode_info(file_path.name)
            
            # Map estimated chunk range to this episode
            for i in range(estimated_chunks + 1):  # +1 for safety
                chunk_to_episode[current_chunk_index + i] = episode_info
            
            current_chunk_index += estimated_chunks + 1
            
            # Copy to temp directory
            temp_file = temp_dir / file_path.name
            temp_file.write_text(content, encoding='utf-8')

        # **NEW: Save episode mapping**
        mapping_file = output_dir / 'episode_mapping.json'
        with open(mapping_file, 'w') as f:
            json.dump(chunk_to_episode, f, indent=2)
        print(f'💾 Saved episode mapping: {mapping_file}')

        # Process the temp directory
        processor.process_directory(temp_dir, output_dir, batch_id="podcast_batch_001")
        
        print(f'✅ Processing complete with episode metadata tracking!')
        
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print('Cleaned up temporary directory')

if __name__ == '__main__':
    main() 