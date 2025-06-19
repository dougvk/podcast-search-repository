#!/usr/bin/env python3

from pathlib import Path
from core.batch_processor import MemvidBatchProcessor, ProcessingConfig
import os
import json
import re
import shutil
import tempfile
import random

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
    # Create config for 20 episodes
    config = ProcessingConfig(max_memory_mb=1024, batch_size=20)
    state_dir = Path('data/batch_state_20')

    # Create processor
    processor = MemvidBatchProcessor(config, state_dir)

    # Process 20 random files
    input_dir = Path('podcast/downloads')
    output_dir = Path('data')

    # Get all txt files
    txt_files = list(input_dir.glob("*.txt"))
    print(f'Found {len(txt_files)} text files in {input_dir}')
    
    if len(txt_files) == 0:
        print("No .txt files found in podcast/downloads")
        return
    
    if len(txt_files) < 20:
        print(f'Only {len(txt_files)} files available, processing all of them')
        files_to_process = txt_files
    else:
        # Randomly select 20 files
        files_to_process = random.sample(txt_files, 20)
    
    print(f'Processing {len(files_to_process)} random episodes for indexing:')
    
    # **NEW: Track episode mapping**
    chunk_to_episode = {}
    current_chunk_index = 0
    
    # Create temporary directory with episode tracking
    temp_dir = Path('temp_podcasts_20')
    temp_dir.mkdir(exist_ok=True)
    
    try:
        for i, file_path in enumerate(files_to_process, 1):
            print(f'  {i:2d}. {file_path.name}')
            
            # Read and track chunks for this episode
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Estimate chunks (rough calculation based on memvid chunking)
            estimated_chunks = max(1, len(content) // 500)  # Rough chunk size estimate
            episode_info = extract_episode_info(file_path.name)
            
            # Map estimated chunk range to this episode
            for j in range(estimated_chunks + 1):  # +1 for safety
                chunk_to_episode[current_chunk_index + j] = episode_info
            
            current_chunk_index += estimated_chunks + 1
            
            # Copy to temp directory
            temp_file = temp_dir / file_path.name
            temp_file.write_text(content, encoding='utf-8')

        # **NEW: Save episode mapping**
        mapping_file = output_dir / 'episode_mapping_20.json'
        with open(mapping_file, 'w') as f:
            json.dump(chunk_to_episode, f, indent=2)
        print(f'💾 Saved episode mapping: {mapping_file} ({len(chunk_to_episode)} entries)')

        # Process the temp directory
        batch_id = "podcast_batch_20_random"
        processor.process_directory(temp_dir, output_dir, batch_id=batch_id)
        
        print(f'✅ Processing complete! Generated:')
        print(f'   - {output_dir}/{batch_id}.mp4 (video)')
        print(f'   - {output_dir}/{batch_id}_index.json (search index)')
        print(f'   - {output_dir}/{batch_id}_index.faiss (embeddings)')
        print(f'   - {output_dir}/episode_mapping_20.json (episode metadata)')
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print('🧹 Cleaned up temporary directory')

if __name__ == '__main__':
    main() 