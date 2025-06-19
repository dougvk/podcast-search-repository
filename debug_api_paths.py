import requests
import json
import os
from pathlib import Path

def debug_api_paths():
    # First, let's check what the current working directory is locally
    print("Local environment:")
    print(f"  Current working directory: {os.getcwd()}")
    print(f"  Episode mapping exists: {Path('data/episode_mapping.json').exists()}")
    print(f"  Episode mapping absolute path: {Path('data/episode_mapping.json').absolute()}")
    print(f"  Index exists: {Path('data/podcast_batch_001_index.json').exists()}")
    print(f"  Index absolute path: {Path('data/podcast_batch_001_index.json').absolute()}")
    
    # Now check what the API can see
    try:
        response = requests.get("http://localhost:8000/api/v1/info")
        if response.status_code == 200:
            info = response.json()
            print(f"\nAPI environment:")
            print(f"  Data directory: {info.get('data_directory')}")
            print(f"  Search engine initialized: {info.get('search_engine_initialized')}")
            print(f"  Available videos: {info.get('available_videos')}")
        else:
            print(f"Failed to get API info: {response.status_code}")
    except Exception as e:
        print(f"Error getting API info: {e}")

if __name__ == "__main__":
    debug_api_paths() 