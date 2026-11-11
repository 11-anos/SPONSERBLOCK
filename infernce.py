import time

import joblib
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    IpBlocked,
    RequestBlocked,
    TranscriptsDisabled,
)


def load_inference_artifacts():
    print("Loading lightweight model artifacts...")
    vectorizer = joblib.load("vectorizer.joblib")
    classifier = joblib.load("classifier.joblib")
    return vectorizer, classifier

def fetch_live_transcript(video_id):
    print(f"Fetching live transcript for video ID: {video_id}...")
    try:
        for attempt in range(3):
            try:
                # Fetch transcript using the current API and convert its objects to dictionaries.
                fetched_data = YouTubeTranscriptApi().fetch(video_id).to_raw_data()
                break
            except requests.exceptions.SSLError:
                if attempt == 2:
                    raise
                print(f"[!] TLS connection failed; retrying ({attempt + 1}/2)...")
                time.sleep(2 ** attempt)

        transcript_chunks = []
        for entry in fetched_data:
            transcript_chunks.append({
                "start": entry.get("start", 0.0),
                "duration": entry.get("duration", 2.0),
                "text": entry.get("text", "")
            })
        return transcript_chunks
        
    except (IpBlocked, RequestBlocked):
        print("\n[!] Error: YouTube flagged this IP or rate-limited the request.")
        return []
    except TranscriptsDisabled:
        print(f"\n[!] Error: Transcripts are disabled for video {video_id}.")
        return []
    except requests.exceptions.SSLError as e:
        print(f"\n[!] TLS/SSL connection failed: {e}")
        print("Check your network, proxy/VPN, system clock, and CA certificates.")
        print("Do not disable SSL verification; that makes the connection insecure.")
        return []
    except Exception as e:  # noqa: BLE001
        print(f"\n[!] Transcript request failed: {e}")
        return []

def detect_sponsors_in_live_video(video_id, threshold=0.3):
    vectorizer, classifier = load_inference_artifacts()
    transcript_chunks = fetch_live_transcript(video_id)
    
    if not transcript_chunks:
        print("[!] Could not retrieve transcript chunks for inference.")
        return []
        
    print(f"Analyzing {len(transcript_chunks)} transcript chunks...")
    texts = [chunk['text'].replace('\n', ' ').strip() for chunk in transcript_chunks]
    
    # Vectorize and predict probabilities
    X_inference = vectorizer.transform(texts)
    probabilities = classifier.predict_proba(X_inference)
    sponsor_probs = probabilities[:, 1]
    
    # Apply tuned threshold
    predictions = (sponsor_probs >= threshold).astype(int)
    
    sponsor_segments = []
    current_segment = None
    
    for i, pred in enumerate(predictions):
        if pred == 1:
            chunk = transcript_chunks[i]
            c_start = chunk['start']
            c_end = c_start + chunk['duration']
            
            if current_segment is None:
                current_segment = {"start": c_start, "end": c_end}
            else:
                current_segment["end"] = c_end
        else:
            if current_segment is not None:
                sponsor_segments.append(current_segment)
                current_segment = None
                
    if current_segment is not None:
        sponsor_segments.append(current_segment)
        
    return sponsor_segments

def dropSegmentsBelowDuration(segments, min_duration=5.0):
    return [seg for seg in segments if seg['end'] - seg['start'] >= min_duration]

if __name__ == "__main__":
    # Test with a valid live video ID (e.g., Linus Tech Tips video ID)
    test_video_id = "IX2NQ1lq4ZM" 
    
    detected_intervals = dropSegmentsBelowDuration(detect_sponsors_in_live_video(test_video_id, threshold=0.7), min_duration=4.0)
    
    print("\n--- Live Detection Results ---")
    if detected_intervals:
        print(f"Found {len(detected_intervals)} sponsor segment(s):")
        for idx, seg in enumerate(detected_intervals, 1):
            print(f"  {idx}. Start: {seg['start']:.2f}s -- End: {seg['end']:.2f}s")
    else:
        print("No sponsor segments detected or video could not be processed.")