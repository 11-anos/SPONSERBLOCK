import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi

def build_dataset_from_bulk(csv_path="/home/sanju/Desktop/SPONSERBLOCK/filtered_sponsors(1).csv", max_videos=50, output_filename="train_dataset.csv"):
    print(f"Loading raw metadata from {csv_path}...")
    
    # 1. Load only necessary columns from the massive bulk CSV to save RAM
    cols_to_keep = ['videoID', 'startTime', 'endTime', 'category', 'votes', 'locked']
    df_sponsors = pd.read_csv(csv_path, usecols=cols_to_keep)
    
    # 2. Filter for valid sponsor segments with positive votes
    sponsor_df = df_sponsors[
        (df_sponsors['category'] == 'selfpromo') & 
        (df_sponsors['votes'] >= 0)
    ]
    
    # Grab a target subset of unique video IDs to process
    video_ids = sponsor_df['videoID'].unique()[:max_videos]
    print(f"Processing {len(video_ids)} unique videos from bulk data...")
    
    all_rows = []
    
    for i, vid in enumerate(video_ids):
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{len(video_ids)} videos processed...")
            
        # Extract sponsor windows for this specific video ID
        vid_rows = sponsor_df[sponsor_df['videoID'] == vid]
        sponsor_windows = list(zip(vid_rows['startTime'], vid_rows['endTime']))
        
        if not sponsor_windows:
            continue
            
        # 3. Fetch YouTube transcript for the video
        try:
            ytt_api = YouTubeTranscriptApi()
            transcript = ytt_api.fetch(vid).to_raw_data()
        except Exception:
            # Skip videos with disabled or unavailable transcripts
            continue
            
        # 4. Align transcript chunks with sponsor time windows
        for chunk in transcript:
            c_start = chunk['start']
            c_duration = chunk['duration']
            c_end = c_start + c_duration
            c_text = chunk['text'].replace('\n', ' ').strip()
            
            if not c_text:
                continue
                
            is_sponsor = 0
            for sp_start, sp_end in sponsor_windows:
                overlap_start = max(c_start, sp_start)
                overlap_end = min(c_end, sp_end)
                overlap_duration = max(0, overlap_end - overlap_start)
                
                # If more than 50% of the chunk overlaps with an ad window, label it 1
                if overlap_duration / c_duration > 0.5:
                    is_sponsor = 1
                    break
                    
            all_rows.append({
                "text": c_text,
                "label": is_sponsor
            })
            
    # 5. Export the clean dataset containing text and label
    if all_rows:
        df_out = pd.DataFrame(all_rows)
        df_out.to_csv(output_filename, index=False, encoding='utf-8')
        print(f"\nSuccessfully compiled {len(df_out)} rows into {output_filename}!")
        print("Label Distribution:")
        print(df_out['label'].value_counts())
    else:
        print("\nNo rows collected. Check your CSV path or video filters.")

if __name__ == "__main__":
    build_dataset_from_bulk()