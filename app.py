from flask import Flask, request, render_template, send_file, jsonify, redirect, url_for, session, flash
import pandas as pd
import pickle
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a secure key

# Simulated user database (Replace with a real database for production)
users = {}

# Paths to model and vectorizer
MODEL_PATH = 'spam.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

# Load model and vectorizer
def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

model = load_pickle(MODEL_PATH)
cv = load_pickle(VECTORIZER_PATH)

if not model or not cv:
    raise RuntimeError("Model or vectorizer failed to load. Ensure 'spam.pkl' and 'vectorizer.pkl' exist.")

STOPWORDS = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    words = word_tokenize(text)
    words = [word for word in words if word not in STOPWORDS]
    return ' '.join(words)

def classify_email(email):
    email = preprocess_text(email)
    vec = cv.transform([email])
    result = model.predict(vec)[0]
    spam_score = model.predict_proba(vec)[0][1] if hasattr(model, "predict_proba") else float(model.decision_function(vec))
    return "Spam" if result == 1 else "Not Spam", round(spam_score, 2)

def process_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'text' not in df.columns:
            return None, "Error: CSV file must have a 'text' column."
        df['text'] = df['text'].astype(str).apply(preprocess_text)
        df[['Prediction', 'Spam Score']] = df['text'].apply(lambda x: pd.Series(classify_email(x)))
        output_path = "classified_emails_sorted.csv"
        df.to_csv(output_path, index=False)
        return output_path, None
    except Exception as e:
        return None, f"Error processing CSV: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html', user=session.get('user'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and check_password_hash(users[username], password):
            session['user'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        flash('Invalid credentials', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            flash('User already exists', 'warning')
            return render_template('register.html')
        users[username] = generate_password_hash(password)
        flash('Registration successful!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/classify-single', methods=['POST'])
def classify_single():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized access. Please log in."}), 401
    email_text = request.form.get('email')
    if not email_text:
        return jsonify({"error": "No email text provided"}), 400
    prediction, score = classify_email(email_text)
    return jsonify({"prediction": prediction, "spam_score": score})

@app.route('/classify', methods=['POST'])
def classify():
    if 'user' not in session:
        return redirect(url_for('login'))
    if 'file' not in request.files:
        flash('No file uploaded', 'danger')
        return redirect(url_for('index'))
    file = request.files['file']
    if not file.filename.endswith('.csv'):
        flash('Only CSV files are allowed', 'danger')
        return redirect(url_for('index'))
    file_path = secure_filename(file.filename)
    file.save(file_path)
    output_path, error = process_csv(file_path)
    if error:
        flash(error, 'danger')
        return redirect(url_for('index'))
    return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
