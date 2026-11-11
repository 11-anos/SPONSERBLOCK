import requests
from youtube_transcript_api import YouTubeTranscriptApi
import json

def fetch_raw_data(video_id):
    # 1. SponsorBlock API
    # Use categories parameter with a JSON array string
    categories = json.dumps(["sponsor"])
    sb_url = f"https://sponsor.ajay.app/api/skipSegments?videoID={video_id}&categories={categories}"
    
    response = requests.get(sb_url)
    sponsor_windows = []
    
    if response.status_code == 200:
        data = response.json()
        sponsor_windows = [segment['segment'] for segment in data]
    elif response.status_code == 404:
        # 404 just means no sponsors found for this video. This is normal.
        pass
    else:
        print(f"Warning: SponsorBlock returned status {response.status_code}")

    # 2. Transcript API (Updated for the new fetch() syntax)
    transcript = []
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id)
    except Exception as e:
        print(f"Transcript failed for {video_id}: {e}")

    return transcript, sponsor_windows

# Test with a video guaranteed to have a sponsor:
test_id = "FfgT6zx4k3Q"# (You can swap this with another known sponsor video)
transcript, sponsor_windows = fetch_raw_data(test_id)

print(f"Found {len(sponsor_windows)} sponsor blocks.")
print(f"Found {len(transcript)} transcript chunks.")