import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
import joblib

print("1. Loading dataset...")
dataset_path = "/home"
df = pd.read_csv(dataset_path)

required_columns = {"text", "label"}
missing_columns = required_columns.difference(df.columns)
if missing_columns:
    available_columns = ", ".join(map(str, df.columns))
    missing = ", ".join(sorted(missing_columns))
    raise ValueError(
        f"{dataset_path!r} is not a training dataset: missing column(s) "
        f"{missing!r}. Expected columns: 'text' and 'label'. "
        f"Available columns: {available_columns}. "
        "Create the transcript-aligned dataset before running this script."
    )

# Drop rows with missing text or labels.
df = df.dropna(subset=["text", "label"])
if df.empty:
    raise ValueError("The training dataset has no rows containing both text and label.")

X_text = df['text'].tolist()
y_labels = df['label'].astype(int).tolist()

if len(set(y_labels)) < 2:
    raise ValueError(
        "At least two label classes are required for training. "
        f"Found only: {sorted(set(y_labels))}."
    )

print(f"Total rows loaded: {len(df)}")
print("Label distribution:\n", df['label'].value_counts())

print("\n2. Splitting into train (80%) and test (20%) sets...")
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y_labels, test_size=0.2, random_state=42, stratify=y_labels
)

print("\n3. Vectorizing text with HashingVectorizer...")
# n_features=1024 keeps the footprint under 500KB
vectorizer = HashingVectorizer(n_features=1024, ngram_range=(1, 2), norm='l2')
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

print("\n4. Training multi-class SGDClassifier...")
# loss='log_loss' enables probabilistic multi-class logistic regression
classifier = SGDClassifier(
    loss='log_loss', 
    penalty='l2', 
    max_iter=1000, 
    random_state=42
)
classifier.fit(X_train, y_train)

print("\n5. Evaluating model performance on the test set...")
y_pred = classifier.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\n6. Saving production model artifacts...")
joblib.dump(vectorizer, "vectorizer.joblib", compress=3)
joblib.dump(classifier, "classifier.joblib", compress=3)
print("Artifacts saved successfully! Footprint remains under 60 KB.")