import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TFSMLayer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import re
import nltk
from nltk.corpus import stopwords
import pickle

nltk.download('stopwords')

# Load the model using TFSMLayer
model_path = "tf_model"  # Path to the SavedModel folder
model = TFSMLayer(model_path, call_endpoint="serving_default")

# Load the tokenizer (ensure tokenizer.pkl is saved alongside the model)
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    '''Removes HTML tags: replaces anything between opening and closing <> with empty space'''
    return TAG_RE.sub('', text)

# Function to preprocess text
def preprocess_text(sen):
    '''Cleans text data up, leaving only 2 or more char long non-stopwords composed of A-Z & a-z only in lowercase'''
    sentence = sen.lower()
    sentence = remove_tags(sentence)
    sentence = re.sub('[^a-zA-Z]', ' ', sentence) 
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence) 
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    sentence = pattern.sub('', sentence)

    return sentence

# Streamlit app
st.title("Sentiment Analysis with LSTM")

st.write("Enter text to analyze its sentiment:")

user_input = st.text_area("Text Input", placeholder="Type your text here...")

if st.button("Analyze Sentiment"):
    unseen_processed = preprocess_text(user_input)
    unseen_tokenized = tokenizer.texts_to_sequences([unseen_processed])  # Ensure input is a list
    unseen_padded = pad_sequences(unseen_tokenized, padding='post', maxlen=100)

    # Cast the input to float32 as expected by the model
    unseen_padded = tf.cast(unseen_padded, tf.float32)

    # Get sentiment from the model (use TFSMLayer for inference)
    unseen_sentiments = model(unseen_padded)  # Model output is a dictionary

    # Extract the sentiment score from the dictionary (the key is 'dense_2')
    sentiment_score = unseen_sentiments['dense_2']  # Use the correct key
    sentiment_score = sentiment_score.numpy()[0][0]  # Assuming single scalar output

    # Convert the output to a sentiment label
    sentiment = "Positive" if sentiment_score > 0.5 else "Negative"
    prediction_score = sentiment_score * 10  # Scale the sentiment score
    st.write(f"**Sentiment Score:** {prediction_score:.2f}")
    st.write(f"**Sentiment:** {sentiment}")
else:
    st.write("Please enter valid text.")
