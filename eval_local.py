import joblib
import pandas as pd

def evaluate_local_samples():
    print("1. Loading model artifacts...")
    vectorizer = joblib.load("vectorizer.joblib")
    classifier = joblib.load("classifier.joblib")
    
    print("2. Loading local training dataset...")
    df = pd.read_csv("train_dataset.csv")
    
    # Grab a small sample of known sponsors (label 1) and content (label 0)
    sponsors = df[df['label'] == 1].head(3)
    content = df[df['label'] == 0].head(3)
    sample_df = pd.concat([sponsors, content])
    
    print(f"\n3. Running local inference on {len(sample_df)} sample chunks...")
    texts = sample_df['text'].tolist()
    actual_labels = sample_df['label'].tolist()
    
    X_inference = vectorizer.transform(texts)
    predictions = classifier.predict(X_inference)
    
    print("\n--- Local Inference Results ---")
    for text, actual, pred in zip(texts, actual_labels, predictions):
        status = "CORRECT" if actual == pred else "MISMATCH"
        print(f"[{status}] Actual: {actual} | Predicted: {pred}")
        print(f"Text: \"{text[:80]}...\"\n")

if __name__ == "__main__":
    evaluate_local_samples()