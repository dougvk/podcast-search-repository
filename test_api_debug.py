import requests
import json

def test_api_episode_mapping():
    # Test a search that should return Boris Johnson content
    response = requests.post(
        "http://localhost:8000/api/v1/search",
        json={"query": "Boris Johnson", "limit": 1}
    )
    
    if response.status_code == 200:
        data = response.json()
        print("API Response:")
        print(json.dumps(data, indent=2))
        
        if data['results']:
            result = data['results'][0]
            episode = result['episode']
            print(f"\nEpisode metadata from API:")
            print(f"  Title: {episode['title']}")
            print(f"  Filename: {episode['filename']}")
            print(f"  Speaker: {episode['speaker']}")
            
            # Check if we got real metadata or placeholder
            if episode['title'] is None and episode['filename'] == 'transcript.txt':
                print("\n❌ Still getting placeholder metadata")
            else:
                print("\n✅ Getting real episode metadata!")
        else:
            print("No results returned")
    else:
        print(f"API request failed: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_api_episode_mapping() 