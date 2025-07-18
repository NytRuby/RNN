# Step 1: Import libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
#import h5py
#import json
import streamlit as st

# Step 2: Load IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Step 3: Load model safely from .h5 without triggering 'batch_shape' error

# Load the fixed model
model = load_model('simple_rnn_imdb.h5')

# Step 4: Helper Functions
# Decode review from integers to words
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Preprocess user input text
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Step 5: Streamlit App UI
st.title('🎬 IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review below, and the model will classify it as **positive** or **negative**.')

user_input = st.text_area('📩 Movie Review Input:')

if st.button('🚀 Classify'):
    if user_input.strip() == "":
        st.warning('Please enter a movie review before clicking "Classify".')
    else:
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)
        sentiment = 'Positive 😊' if prediction[0][0] > 0.5 else 'Negative 😞'
        st.markdown(f'**Sentiment:** {sentiment}')
        st.markdown(f'**Prediction Score:** {prediction[0][0]:.4f}')
else:
    st.info('Awaiting input...')

