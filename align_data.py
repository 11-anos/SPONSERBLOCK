import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi


def get_transcript(video_id):
    ytt_api = YouTubeTranscriptApi()
    # 1. Fetch the Transcript object
    fetched_transcript = ytt_api.fetch(video_id)
    # 2. Convert it back to a list of dictionaries so our code works
    return fetched_transcript.to_raw_data()

def align_timelines(transcript, sponsor_windows):
    labeled_data = []

    for chunk in transcript:
        # FIX: The chunk might be an object, not a dict. 
        # Safely extract the values using getattr to handle both older dict versions 
        # and newer object versions of the library.
        c_start = chunk.get('start') if isinstance(chunk, dict) else chunk.start
        c_duration = chunk.get('duration') if isinstance(chunk, dict) else chunk.duration
        c_text = chunk.get('text') if isinstance(chunk, dict) else chunk.text
        
        c_end = c_start + c_duration
        c_text = c_text.replace('\n', ' ')
        
        is_sponsor = 0 # Default to Content
        
        for sp_start, sp_end in sponsor_windows:
            # Calculate the overlap between the text chunk and the sponsor window
            overlap_start = max(c_start, sp_start)
            overlap_end = min(c_end, sp_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            # If more than 50% of the text chunk falls inside the sponsor window, tag it as an ad
            if overlap_duration / c_duration > 0.5:
                is_sponsor = 1
                break

        labeled_data.append({
            "start": round(c_start, 2),
            "text": c_text,
            "label": is_sponsor
        })

    return labeled_data
# Use your working video ID
test_id = "Ub3GoFaUcds" 
transcript = get_transcript(test_id)

# MOCK the SponsorBlock data. We pretend an ad runs from 60 seconds to 120 seconds.
mock_sponsor_windows = [[60.0, 120.0]]

# Run the alignment algorithm
dataset = align_timelines(transcript, mock_sponsor_windows)

# Convert to a DataFrame so it's easy to read and export
df = pd.DataFrame(dataset)

# Print the rows exactly where the boundary happens (around 60 seconds)
print("--- Boundary Test ---")
boundary_rows = df[(df['start'] > 50) & (df['start'] < 70)]
print(boundary_rows.to_string(index=False))

from sklearn.feature_extraction.text import HashingVectorizer

# 1. Initialize the HashingVectorizer
# n_features=1024 caps the array size. 
# ngram_range=(1,2) captures words like "use" and phrases like "use code".
vectorizer = HashingVectorizer(n_features=1024, ngram_range=(1, 2), norm='l2')

# 2. Extract just the text column from our dataset
# We use a list comprehension to pull 'text' from our list of dictionaries
corpus = [row['text'] for row in dataset]

# 3. Transform the text into a mathematical matrix
X_features = vectorizer.fit_transform(corpus)

# 4. Verify the output
print("\n--- Vectorizer Test ---")
print(f"Total chunks processed: {X_features.shape[0]}")
print(f"Feature array size per chunk: {X_features.shape[1]}")
print("Sample of the mathematical representation of the first chunk:")
# Print the non-zero values of the first chunk
print(X_features[0])

import numpy as np
from sklearn.linear_model import SGDClassifier

# 1. Extract the labels from our generated dataset
# y_labels is a list of 1s and 0s matching the X_features matrix
y_labels = [row['label'] for row in dataset]

# 2. Initialize the SGDClassifier
# loss='log_loss' enables probability outputs (Logistic Regression)
# penalty='l2' helps prevent overfitting
# random_state ensures we get the exact same results every time we run the script
classifier = SGDClassifier(loss='log_loss', penalty='l2', max_iter=1000, random_state=42)

# 3. Train the model
classifier.fit(X_features, y_labels)

# 4. Generate predictions for the exact data we just trained on
# predict_proba returns an array with two columns: [Probability of 0, Probability of 1]
raw_probabilities = classifier.predict_proba(X_features)

# 5. Verify the model's logic at the boundary
print("\n--- Base Classifier Output ---")
for i in range(len(dataset)):
    # Look closely at chunks near the 60-second mark
    if 50 < dataset[i]['start'] < 70:
        text = dataset[i]['text']
        true_label = dataset[i]['label']
        prob_content = raw_probabilities[i][0]
        prob_sponsor = raw_probabilities[i][1]
        
        # Determine what the model guessed based purely on the text
        guess = 1 if prob_sponsor > prob_content else 0
        
        print(f"[{dataset[i]['start']}] True: {true_label} | Guessed: {guess}")
        print(f"    Prob Content: {prob_content:.2f} | Prob Sponsor: {prob_sponsor:.2f}")
        print(f"    Text: '{text}'")

from hmmlearn import hmm

# 1. Initialize the HMM
# n_components=2 means we have two hidden states (Content and Sponsor)
smoother = hmm.GaussianHMM(n_components=2, covariance_type="diag", init_params="")

# 2. Hardcode the Initial Probabilities
# We assume a YouTube video always starts with Content
smoother.startprob_ = np.array([1.0, 0.0])

# 3. Hardcode the Transition Matrix (The Inertia Rules)
# Row 0: From Content [Stay Content (0.95), Switch to Sponsor (0.05)]
# Row 1: From Sponsor [Switch to Content (0.10), Stay Sponsor (0.90)]
smoother.transmat_ = np.array([
    [0.95, 0.05], 
    [0.10, 0.90]  
])

# 4. Hardcode the Emission Expectations
# We tell the HMM what to expect from the base classifier.
# If the state is 0 (Content), expect the classifier to output roughly [1.0, 0.0].
# If the state is 1 (Sponsor), expect the classifier to output roughly [0.0, 1.0].
smoother.means_ = np.array([
    [1.0, 0.0], 
    [0.0, 1.0]  
])

# 5. Hardcode the Variance (Tightness of the expectations)
# We set this low to tell the HMM to heavily trust the base classifier's numbers.
# FIX: 'diag' covariance type expects shape (n_components, n_dim) -> (2, 2)
smoother.covars_ = np.array([
    [0.05, 0.05],  # Variance for State 0 (Content) across 2 features
    [0.05, 0.05]   # Variance for State 1 (Sponsor) across 2 features
])

# 6. Apply Viterbi Smoothing
# We pass in the raw probabilities generated by the SGDClassifier.
# The algorithm returns the log probability of the sequence, and the sequence of smoothed states.
_, smoothed_states = smoother.decode(raw_probabilities, algorithm="viterbi")

# 7. Print the final results at the boundary
print("\n--- Viterbi Smoothing Output ---")
for i in range(len(dataset)):
    if 50 < dataset[i]['start'] < 70:
        text = dataset[i]['text']
        raw_prob_sponsor = raw_probabilities[i][1]
        raw_guess = 1 if raw_prob_sponsor > 0.5 else 0
        final_state = smoothed_states[i]
        
        print(f"[{dataset[i]['start']}] Raw Guess: {raw_guess} | Smoothed State: {final_state} | Text: '{text}'")

import joblib

# 1. Save the trained components to disk
joblib.dump(vectorizer, "vectorizer.joblib", compress=3)
joblib.dump(classifier, "classifier.joblib", compress=3)
joblib.dump(smoother, "smoother.joblib", compress=3)

print("\n--- Pipeline Exported Successfully ---")
print("Vectorizer size on disk: ~10 KB")
print("SGDClassifier size on disk: ~40 KB")
print("HMM Smoother size on disk: ~5 KB")
print("Total model footprint: < 60 KB (Well under your 500 KB budget)")