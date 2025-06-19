#!/usr/bin/env python3

import json
from pathlib import Path

def extract_frame_from_text(text):
    """Extract frame number by looking up text in memvid index"""
    try:
        index_file = Path("data/podcast_batch_001_index.json")
        if index_file.exists():
            with open(index_file, 'r') as f:
                index_data = json.load(f)
            
            # Debug: Check the structure
            print(f"Index has keys: {list(index_data.keys())}")
            
            # Search for matching text in index metadata
            if 'metadata' in index_data:
                print(f"Found {len(index_data['metadata'])} items in metadata")
                
                # Try to match the text
                for i, item in enumerate(index_data['metadata']):
                    item_text = item.get('text', '').strip()
                    if len(item_text) > 100:
                        search_text = text.strip()[:100]
                        if item_text.startswith(search_text):
                            print(f"✅ MATCH! Frame {item.get('frame')} for item {i}")
                            return item.get('frame', 0)
                        elif i < 5:  # Show first few for debugging
                            print(f"❌ No match for item {i}: '{item_text[:50]}...'")
                
                print(f"❌ No matches found for search text: '{text[:100]}...'")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 0

# Test with actual Boris Johnson search result
test_text = """nister. He's trying to cling on. By the time you listen to this, you may have
 gone as prime minister as well. So Dominic, you wrote a fabulous essay for UnHerd, U-N-H-E-R-D,
 a couple days ago, on Boris Johnson as a political leader."""

print("Testing frame extraction...")
frame = extract_frame_from_text(test_text)
print(f"Extracted frame: {frame}")

# Also check the episode mapping
mapping_file = Path("data/episode_mapping.json")
if mapping_file.exists():
    with open(mapping_file, 'r') as f:
        episode_mapping = json.load(f)
    print(f"\nEpisode mapping has {len(episode_mapping)} entries")
    print(f"Frame {frame} maps to: {episode_mapping.get(str(frame), 'NOT FOUND')}") 