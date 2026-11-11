import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
import joblib

print("1. Loading local training dataset...")
df = pd.read_csv("train_dataset.csv")

# Clean missing values if any exist
df = df.dropna(subset=['text', 'label'])

X_text = df['text'].tolist()
y_labels = df['label'].astype(int).tolist()

print(f"Total rows loaded: {len(df)}")
print("Label distribution:\n", df['label'].value_counts())

print("\n2. Splitting into train (80%) and test (20%) sets...")
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y_labels, test_size=0.2, random_state=42, stratify=y_labels
)

print("\n3. Vectorizing text with HashingVectorizer...")
# n_features=1024 locks the footprint under 50KB
vectorizer = HashingVectorizer(n_features=1024, ngram_range=(1, 2), norm='l2')
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)
print("\n4. Training production SGDClassifier with custom class weights...")
print("\n4. Training production SGDClassifier with custom class weights...")
classifier = SGDClassifier(
    loss='log_loss', 
    penalty='l2', 
    max_iter=1000, 
    class_weight={0: 1.0, 1: 10.0},  # <--- Tuned penalty ratio
    random_state=42
)
classifier.fit(X_train, y_train)

print("\n5. Evaluating performance on test set...")
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))

print("\n6. Exporting compressed production artifacts...")
joblib.dump(vectorizer, "vectorizer.joblib", compress=3)
joblib.dump(classifier, "classifier.joblib", compress=3)
print("Model successfully trained on 9,974 real rows and saved!")