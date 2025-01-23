# Import necessary libraries
import os
import re
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import streamlit as st

from transformers import pipeline

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

# API Configuration
API_KEY = "AIzaSyAWtj8F33sTl04ln4jr6uIRD8Xd_D-fsME"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# Function to fetch comments from YouTube
def fetch_comments(video_id, max_results=100):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)
    comments = []
    response = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        textFormat="plainText"
    ).execute()

    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        comments.append(comment)

    return comments

# Function to preprocess comments
def preprocess_comments(comments):
    stop_words = set(stopwords.words('english'))
    preprocessed = []
    for comment in comments:
        # Remove special characters and convert to lowercase
        comment = re.sub(r'[^a-zA-Z\s]', '', comment).lower()
        # Tokenize and remove stopwords
        tokens = nltk.word_tokenize(comment)
        filtered_tokens = [word for word in tokens if word not in stop_words]
        preprocessed.append(' '.join(filtered_tokens))
    return preprocessed

# Function for sentiment analysis
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

def analyze_sentiments(comments):
    sentiment_model = load_sentiment_model()
    results = sentiment_model(comments)
    sentiments = [result['label'] for result in results]
    return sentiments

# Visualization Functions
def plot_sentiment_distribution(sentiments):
    sentiment_counts = pd.Series(sentiments).value_counts()
    plt.figure(figsize=(6, 6))
    sentiment_counts.plot.pie(autopct="%1.1f%%", colors=['#66c2a5', '#fc8d62', '#8da0cb'])
    plt.title("Sentiment Distribution")
    st.pyplot(plt)

# Word Cloud Generation
def generate_wordcloud(comments):
    text = ' '.join(comments)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Streamlit GUI
st.title("YouTube Comment Sentiment Analyzer")
st.write("Enter a YouTube video ID to analyze comments for sentiment.")

video_id = st.text_input("YouTube Video ID", placeholder="Enter video ID here...")

if st.button("Analyze"):
    if video_id:
        with st.spinner("Fetching comments..."):
            comments = fetch_comments(video_id)

        if comments:
            st.success(f"Fetched {len(comments)} comments!")

            with st.spinner("Processing comments..."):
                preprocessed_comments = preprocess_comments(comments)

            with st.spinner("Performing sentiment analysis..."):
                sentiments = analyze_sentiments(preprocessed_comments)

            st.subheader("Sentiment Distribution")
            plot_sentiment_distribution(sentiments)

            st.subheader("Word Cloud")
            generate_wordcloud(preprocessed_comments)

            st.subheader("Sample Comments and Sentiments")
            for comment, sentiment in zip(comments[:10], sentiments[:10]):
                st.write(f"- **Comment**: {comment}")
                st.write(f"  **Sentiment**: {sentiment}")
                st.write("---")
        else:
            st.error("No comments found for this video.")
    else:
        st.error("Please enter a valid YouTube video ID.")