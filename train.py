import joblib
import pandas as pd
import requests
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from youtube_transcript_api import YouTubeTranscriptApi


def fetch_real_sponsor_data(video_ids):
    all_dataset_rows = []
    
    for vid in video_ids:
        print(f"Processing video: {vid}...")
        
        # 1. Pull real SponsorBlock segments
        sb_url = f"https://sponsor.ajay.app/api/skipSegments?videoID={vid}"
        response = requests.get(sb_url)
        sponsor_windows = []
        
        if response.status_code == 200:
            data = response.json()
            # Extract real sponsor windows
            sponsor_windows = [item['segment'] for item in data if item.get('category') == 'sponsor']
        elif response.status_code == 404:
            # 404 means no sponsors for this video, skip it safely
            continue
        else:
            continue

        # 2. Pull real transcript
        try:
            ytt_api = YouTubeTranscriptApi()
            transcript = ytt_api.fetch(vid).to_raw_data()
        except Exception:  # noqa: BLE001, S112
            # If transcript fails or video is unavailable, skip
            continue

        # 3. Align real transcript chunks with real sponsor windows
        for chunk in transcript:
            c_start = chunk['start']
            c_duration = chunk['duration']
            c_end = c_start + c_duration
            c_text = chunk['text'].replace('\n', ' ')
            
            is_sponsor = 0
            for sp_start, sp_end in sponsor_windows:
                overlap_start = max(c_start, sp_start)
                overlap_end = min(c_end, sp_end)
                overlap_duration = max(0, overlap_end - overlap_start)
                
                if overlap_duration / c_duration > 0.5:
                    is_sponsor = 1
                    break

            all_dataset_rows.append({
                "text": c_text,
                "label": is_sponsor
            })
            
    return all_dataset_rows
# Add this check right after your fetch loop in train_production.py

# --- Execution ---
# Provide a small list of verified video IDs with real sponsors
training_video_ids = ["FfgT6zx4k3Q", "IqJoMdxT7XI"] 

print("Building dataset from real SponsorBlock data...")
master_dataset = fetch_real_sponsor_data(training_video_ids)

print(f"Total rows collected: {len(master_dataset)}")
if len(master_dataset) > 0:
    df = pd.DataFrame(master_dataset)
    print("Label distribution:")
    print(df['label'].value_counts())

if len(master_dataset) > 0:
    df = pd.DataFrame(master_dataset)
    print(f"Successfully built dataset with {len(df)} total rows.")
    
    # Extract text and labels for training
    corpus = df['text'].tolist()
    y_labels = df['label'].tolist()
    
    # Vectorize and Train
    vectorizer = HashingVectorizer(n_features=1024, ngram_range=(1, 2), norm='l2')
    X_train = vectorizer.fit_transform(corpus)
    
    classifier = SGDClassifier(loss='log_loss', penalty='l2', max_iter=1000, random_state=42)
    classifier.fit(X_train, y_labels)
    
    print("Model successfully trained on real crowdsourced sponsor patterns!")
else:
    print("No valid data collected. Check video IDs or network connection.")
    

# Save the newly trained real-data artifacts
joblib.dump(vectorizer, "vectorizer.joblib", compress=3)
joblib.dump(classifier, "classifier.joblib", compress=3)

print("\n--- Production Artifacts Updated Successfully ---")