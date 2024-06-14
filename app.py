import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import re
import nltk
import string
import pickle
import numpy as np
import sqlite3
import tensorflow as tf
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from num2words import num2words

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from collections import defaultdict
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from flask import Flask, jsonify
from flask import request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from
from keras.models import load_model


factory = StemmerFactory()
stemmer = factory.create_stemmer()
list_stopwords = set(stopwords.words('indonesian'))

# Load your model and tokenizer
model_nn = load_model('nn_model.h5')
# Memuat model LSTM
model_lstm = load_model('lstm_model.h5')

# Memuat tokenizer (jika belum dimuat di bagian lain)
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

class_labels = ['negatif', 'netral', 'positif']

# Function for text preprocessing
def cleansing(text):
    text = re.sub(r'\\t|\\n|\\u', ' ', text)
    text = re.sub(r"https?:[^\s]+", ' ', text)
    text = re.sub(r'(\b\w+)-\1\b', r'\1', text)
    text = re.sub(r'[\\x]+[a-z0-9]{2}', '', text)
    text = re.sub(r'[^a-zA-Z]+', ' ', text)
    text = re.sub(r'\brt\b|\buser\b', ' ', text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in list_stopwords]
    tokens = [stemmer.stem(word) for word in tokens]
    text = ' '.join(tokens)
    return text

app = Flask(__name__)

app.json_encoder = LazyJSONEncoder
swagger_template = dict(
info = {
    'title': 'API documentation for ML and DL',
    'version': '1.0.1',
    'description': 'API for sentiment prediction using keras NN, LSTM, and MLPClassifier models',
    },
#     host = "127.0.0.1:5000"
#     host = LazyString(lambda: request.host)
)

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json',
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}

swagger = Swagger(app, template=swagger_template,             
                  config=swagger_config)

@swag_from("docs/hello_world.yml", methods=['GET'])
@app.route('/', methods=['GET'])
def hello_world():
    json_response = {
        'Platinum Challenge': "Tweet Sentiment Analysis using keras NN, LSTM, and MLPClassifier models",
    }

    response_data = jsonify(json_response)
    return response_data

@swag_from('docs/nn.yaml', methods=['POST'])
@app.route('/nn', methods=['POST'])
def nn():
    original_text = request.form.get('text')
    cleaned_text = cleansing(original_text)
    text_sequence = tokenizer.texts_to_sequences([cleaned_text])
    text_padded = pad_sequences(text_sequence, maxlen=64)  # Sesuaikan maxlen sesuai kebutuhan Anda
    prediction = model_nn.predict(text_padded)
    predicted_class = np.argmax(prediction[0])
    get_sentiment = class_labels[predicted_class]

    json_response = {
        'status_code': 200,
        'description': "NN Prediction Result",
        'data': {
            'text': original_text,
            'sentiment': get_sentiment,
        }
    }
    return jsonify(json_response)

# Define the /lstm route
@swag_from('docs/lstm.yaml', methods=['POST'])
@app.route('/lstm', methods=['POST'])
def lstm():
    original_text = request.form.get('text')
    cleaned_text = cleansing(original_text)
    text_sequence = tokenizer.texts_to_sequences([cleaned_text])
    text_padded = pad_sequences(text_sequence, maxlen=64)  # Sesuaikan maxlen sesuai kebutuhan Anda
    prediction = model_lstm.predict(text_padded)
    predicted_class = np.argmax(prediction[0])
    get_sentiment = class_labels[predicted_class]

    json_response = {
        'status_code': 200,
        'description': "LSTM Prediction Result",
        'data': {
            'text': original_text,
            'sentiment': get_sentiment,
        }
    }
    return jsonify(json_response)

if __name__ == '__main__':
   app.run()