"""
Simple SMS Spam Classifier
- Loads SMS spam dataset
- Trains TF-IDF + Naive Bayes classifier
- Interactive prediction loop
"""

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset from CSV (safe column handling)
df = pd.read_csv('spam.csv', encoding='latin-1', usecols=[0, 1], header=0, names=['v1', 'v2'], dtype=str)
df = df.rename(columns={'v1': 'label', 'v2': 'text'})
df = df.dropna(subset=['label', 'text']).reset_index(drop=True)

# Normalize labels and convert to binary (ham=0, spam=1)
df['label'] = df['label'].str.strip().str.lower().map({'ham': 0, 'spam': 1})
df = df[df['label'].isin([0, 1])].reset_index(drop=True)

print(f"Dataset: {len(df)} messages (spam={int(df['label'].sum())}, ham={len(df) - int(df['label'].sum())})")

# Simple text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)   # remove URLs
    text = re.sub(r'\S+@\S+', ' ', text)           # remove emails
    text = re.sub(r'[^a-z0-9\s]', ' ', text)       # keep alphanum only
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['text_clean'] = df['text'].apply(clean_text)

# Split data (use stratify to keep label distribution)
X_train, X_test, y_train, y_test = train_test_split(
    df['text_clean'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# Build and train model (small, sensible defaults)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
    ('nb', MultinomialNB(alpha=0.1))
])

pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['ham', 'spam'], zero_division=0))

# Interactive prediction
def predict_message(text):
    cleaned = clean_text(text)
    label = pipeline.predict([cleaned])[0]
    prob = pipeline.predict_proba([cleaned])[0][1]
    return 'spam' if label == 1 else 'ham', prob

print("\n--- Interactive Spam Checker ---")
print("Type a message or 'quit' to exit")

while True:
    msg = input("\nEnter message > ").strip()
    if msg.lower() in ['quit', 'exit']:
        print("Goodbye!")
        break

    label, prob = predict_message(msg)
    print(f"=> {label.upper()} (spam probability: {prob:.4f})")
