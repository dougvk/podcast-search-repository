import json
from pathlib import Path

def test_episode_mapping():
    # Load episode mapping
    mapping_file = Path('data/episode_mapping.json')
    if mapping_file.exists():
        with open(mapping_file, 'r') as f:
            episode_mapping = json.load(f)
        print(f'Episode mapping loaded: {len(episode_mapping)} entries')
        print('Sample entries:')
        for i, (k, v) in enumerate(episode_mapping.items()):
            if i < 3:
                print(f'  {k}: {v}')
    else:
        print('Episode mapping file not found')
        return

    # Load index data
    index_file = Path('data/podcast_batch_001_index.json')
    if index_file.exists():
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        print(f'\nIndex loaded with {len(index_data.get("metadata", []))} metadata entries')
        
        # Search for Boris Johnson text
        boris_text = "nister. He's trying to cling on. By the time you listen to this, you may have"
        found_frames = []
        for item in index_data['metadata']:
            text = item.get('text', '')
            if 'Boris Johnson' in text:
                frame = item.get('frame', 0)
                found_frames.append(frame)
                print(f'  Found Boris Johnson at frame {frame}')
                
                # Check if frame has episode mapping
                episode_data = episode_mapping.get(str(frame))
                if episode_data:
                    print(f'    -> Episode: {episode_data.get("title", "Unknown")}')
                else:
                    print(f'    -> No episode mapping for frame {frame}')
        
        print(f'\nFound {len(found_frames)} frames with Boris Johnson content')
        
        # Test frame extraction logic
        def extract_frame_from_text(text):
            """Extract frame number by looking up text in memvid index"""
            try:
                if 'metadata' in index_data:
                    for item in index_data['metadata']:
                        if item.get('text', '').strip().startswith(text.strip()[:100]):
                            return item.get('frame', 0)
                return 0
            except Exception:
                return 0
        
        print(f'\nTesting frame extraction with sample Boris Johnson text...')
        test_text = boris_text
        extracted_frame = extract_frame_from_text(test_text)
        print(f'Extracted frame: {extracted_frame}')
        
        if extracted_frame > 0:
            episode_data = episode_mapping.get(str(extracted_frame))
            if episode_data:
                print(f'Episode metadata: {episode_data}')
            else:
                print(f'No episode mapping for extracted frame {extracted_frame}')
    else:
        print('Index file not found')

if __name__ == "__main__":
    test_episode_mapping() 