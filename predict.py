import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Load model & vectorizer
with open('spam.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

# Function to predict a new email
def predict_email(email):
    email = preprocess_text(email)  # Clean email
    email_vec = vectorizer.transform([email])  # Convert to numerical format
    prediction = model.predict(email_vec)[0]  # Predict
    return "Spam" if prediction == 1 else "Not Spam"

# Example usage
email = input("📩 Enter an email: ")
print(f"🔍 Prediction: {predict_email(email)}")
