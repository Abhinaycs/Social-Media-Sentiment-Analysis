

---

## Social Media Sentiment Analysis

### Project Overview

This project aims to analyze sentiment from social media posts using natural language processing (NLP) techniques and logistic regression. The objective is to assess public sentiment on various topics by processing and classifying text data collected from Twitter. The project includes data collection, preprocessing, sentiment classification, visualization, and interactive deployment.

### Technologies Used

- **Data Collection**: Tweepy API
- **Text Preprocessing**: NLTK, spaCy
- **Sentiment Classification**: Logistic Regression
- **Visualization**: Matplotlib
- **Deployment**: Streamlit

### 1. Data Collection

**API**: The project uses the Tweepy API to collect tweets related to specific keywords or topics. Tweepy allows for easy access to Twitter data, including tweets, user information, and metadata.

**Steps**:
- Set up a Twitter Developer account and obtain API credentials.
- Use Tweepy to authenticate and access the Twitter API.
- Collect tweets based on predefined search criteria or keywords.

### 2. Text Preprocessing

**Libraries**: NLTK and spaCy are used for text preprocessing tasks.

**Steps**:
- **Tokenization**: Split text into individual words or tokens.
- **Lowercasing**: Convert all text to lowercase to ensure uniformity.
- **Removing Punctuation and Stop Words**: Clean the text by removing punctuation and common stop words that do not contribute to sentiment.
- **Stemming**: Reduce words to their root form using NLTK’s Porter Stemmer.

**Code Example**:
```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from spacy.lang.en import English

# Initialize the stemmer and spaCy tokenizer
port_stem = PorterStemmer()
tokenizer = English().Defaults.create_tokenizer()

def preprocess_text(text):
    # Remove non-alphabetic characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    tokens = tokenizer(text)
    tokens = [port_stem.stem(token.text) for token in tokens if token.text not in stopwords.words('english')]
    return ' '.join(tokens)
```

### 3. Sentiment Classification

**Algorithm**: Logistic Regression is used for sentiment classification. It is a binary classification model that predicts sentiment labels (e.g., positive or negative) based on the processed text data.

**Steps**:
- **Feature Extraction**: Convert text into numerical features using techniques such as TF-IDF or Count Vectorization.
- **Model Training**: Train the logistic regression model on labeled sentiment data.
- **Prediction**: Use the trained model to classify the sentiment of new, unseen tweets.

**Code Example**:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Vectorize the text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(tweets['text'])
y = tweets['sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

### 4. Visualization

**Library**: Matplotlib is used to visualize the sentiment analysis results.

**Steps**:
- **Sentiment Distribution**: Create bar charts or pie charts to show the distribution of sentiments (positive, negative).
- **Trend Analysis**: Plot trends over time or by specific topics.

**Code Example**:
```python
import matplotlib.pyplot as plt

# Example sentiment distribution plot
sentiment_counts = tweets['sentiment'].value_counts()
plt.bar(sentiment_counts.index, sentiment_counts.values)
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.title('Sentiment Distribution')
plt.show()
```

### 5. Deployment

**Platform**: Streamlit is used to deploy the sentiment analysis model interactively.

**Steps**:
- **Build the Streamlit App**: Create an interface for users to input text and view sentiment analysis results in real-time.
- **Run the Application**: Deploy the app on a local server or cloud platform.

**Code Example**:
```python
import streamlit as st

# Streamlit app
st.title('Social Media Sentiment Analysis')
text_input = st.text_area('Enter text here:')
if text_input:
    processed_text = preprocess_text(text_input)
    features = vectorizer.transform([processed_text])
    sentiment = model.predict(features)[0]
    st.write(f'Sentiment: {"Positive" if sentiment == 1 else "Negative"}')
```

### Conclusion

This project demonstrates a comprehensive approach to analyzing social media sentiment using NLP and machine learning. By leveraging Tweepy for data collection, NLTK and spaCy for preprocessing, logistic regression for classification, and Streamlit for deployment, the project provides a robust tool for understanding public sentiment on various topics.
