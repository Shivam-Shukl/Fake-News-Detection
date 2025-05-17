from flask import Flask, request, render_template, flash, redirect, url_for
import pickle
import re
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flashing messages

# Load model and vectorizer safely
MODEL_PATH = 'model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

def load_pickle(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    with open(path, 'rb') as f:
        return pickle.load(f)

model = load_pickle(MODEL_PATH)
vectorizer = load_pickle(VECTORIZER_PATH)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    news = ""
    if request.method == 'POST':
        news = request.form.get('news', '')
        if not news.strip():
            flash("Please enter news text to analyze.", "warning")
            return redirect(url_for('home'))
        clean_news = clean_text(news)
        transformed = vectorizer.transform([clean_news])
        pred = model.predict(transformed)[0]
        print(app.url_map)
        prediction = "Fake News" if pred == 1 else "Real News"
    return render_template('index.html', prediction=prediction, news=news)

if __name__ == '__main__':
    app.run(debug=True)