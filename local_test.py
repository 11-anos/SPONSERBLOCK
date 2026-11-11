import joblib
import pandas as pd

print("1. Loading local model artifacts...")
vectorizer = joblib.load("vectorizer.joblib")
classifier = joblib.load("classifier.joblib")

print("2. Loading your compiled training dataset...")
df = pd.read_csv("train_dataset.csv")

# Grab a few known ad chunks (label 1) and regular text chunks (label 0)
sample_df = pd.concat([
    df[df['label'] == 1].head(3),
    df[df['label'] == 0].head(3)
])

print("\n3. Running local inference test...")
texts = sample_df['text'].tolist()
actuals = sample_df['label'].tolist()

X = vectorizer.transform(texts)
preds = classifier.predict(X)

for text, actual, pred in zip(texts, actuals, preds):
    status = "SUCCESS (Correct)" if actual == pred else "MISMATCH"
    print(f"\n[{status}] Actual Label: {actual} | Predicted: {pred}")
    print(f"Text snippet: \"{text}\"")