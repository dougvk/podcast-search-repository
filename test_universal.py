#!/usr/bin/env python3
from universal_api import discover_data

try:
    video_file, index_file, mapping_file = discover_data()
    print(f"Video: {video_file}")
    print(f"Index: {index_file}")
    print(f"Mapping: {mapping_file}")
    print(f"Video exists: {video_file.exists()}")
    print(f"Index exists: {index_file.exists()}")
    print(f"Mapping exists: {mapping_file.exists() if mapping_file else 'None'}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 