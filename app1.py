import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords')

# Load the pre-trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Initialize stemmer
port_stem = PorterStemmer()

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    tokens = text.split()
    tokens = [port_stem.stem(word) for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

st.title('Social Media Sentiment Analysis')
text_input = st.text_area('Enter text here:')
if text_input:
    processed_text = preprocess_text(text_input)
    features = vectorizer.transform([processed_text])
    sentiment = model.predict(features)[0]
    st.write(f'Sentiment: {"Positive" if sentiment == 1 else "Negative"}')
