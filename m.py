import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
from sklearn.metrics import accuracy_score

# Load the saved model and vectorizer
with open('spam.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Preprocessing function (Same as in train_model.py)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

# Load dataset for testing
df = pd.read_csv("emails.csv")
df['cleaned_text'] = df['text'].apply(preprocess_text)

# Transform text into numerical format
X_test = vectorizer.transform(df['cleaned_text'])

# Predict spam or not spam
df['prediction'] = model.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(df['spam'], df['prediction'])
print(f"✅ Model Accuracy: {accuracy:.2f}")

# Show misclassified emails
misclassified = df[df['spam'] != df['prediction']]
print("🚨 Misclassified Emails:")
print(misclassified[['text', 'spam', 'prediction']])
