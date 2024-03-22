from flask import Flask, request, render_template
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__, template_folder='templates')


# Load the model from the specified path
model = joblib.load('model.pkl')



# Load the fitted TF-IDF vectorizer
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the review text from the form 
    review = request.form['review']

    # Transform the review text using the loaded vectorizer
    review_vectorized = vectorizer.transform([review])

    # Predict sentiment
    sentiment = model.predict(review_vectorized)[0]

    return render_template('index.html', sentiment=sentiment, review=review)

if __name__ == '__main__':
    app.run(debug=True)
