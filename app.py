from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import tensorflow as tf
import pickle

app = Flask(__name__)

# Load the models
model_nn = load_model('Neural Network.h5')
model_lstm = load_model('model_lstm.h5')

# Load the tokenizer
with open('/Users/admin/Documents/Platinum Challenge/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

# Preprocess function
def preprocess_text(text):
    stop_words = set(stopwords.words('indonesian'))
    text = text.lower()                                 # Mengubah menjadi huruf kecil
    text = re.sub(r'\s+', ' ', text)                    # Mengganti banyak spasi dengan satu spasi
    text = re.sub(r'[^a-zA-Z\s]', '', text)             # Menghapus karakter spesial kecuali spasi
    text = re.sub(r'\t', ' ', text)                     # Menghapus tab dan menggantinya dengan spasi
    text = re.sub(r'\d+', '', text)                     # Menghapus angka
    text = re.sub(f"[{string.punctuation}]", "", text)  # Menghapus tanda baca
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Menghapus stopwords
    return text

@app.route('/train', methods=['POST'])
def train_model():
    # Load Dataset
    data = pd.read_csv('/train_preprocess.tsv.txt', sep='\t', header=None, names=['text', 'label'])

    # Preprocess Data
    data['cleaned_text'] = data['text'].apply(preprocess_text)
    sequences = tokenizer.texts_to_sequences(data['cleaned_text'])
    maxlen = max([len(seq) for seq in sequences])
    X = pad_sequences(sequences, maxlen=maxlen)
    y = data['label'].values

    # Determine input_dim and set output_dim, vocab_size
    input_dim = len(tokenizer.word_index) + 1
    output_dim = 128
    vocab_size = len(tokenizer.word_index) + 1

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

    # Fit and transform labels to one-hot encoded format
    onehot_encoder = OneHotEncoder()
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    y_train_hot = onehot_encoder.fit_transform(y_train).toarray()
    y_test_hot = onehot_encoder.transform(y_test).toarray()

    # Train the model
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=80, restore_best_weights=True)
    history = model_lstm.fit(X_train, y_train_hot, epochs=100, validation_data=(X_test, y_test_hot), batch_size=64, callbacks=[es])
    return jsonify({'message': 'Model trained successfully!'})

@app.route('/evaluate', methods=['POST'])
def evaluate_model():
    # Get the test data
    test_data = request.json['test_data']
    sequences = tokenizer.texts_to_sequences(test_data)
    maxlen = max([len(seq) for seq in sequences])
    X = pad_sequences(sequences, maxlen=maxlen)

    # Get the true labels
    y_true_lstm = np.array(request.json['y_true_lstm'])

    # Evaluate the model
    loss, accuracy = model_lstm.evaluate(X, y_true_lstm)

    return jsonify({'loss': loss, 'accuracy': accuracy})

@app.route('/predict_lstm', methods=['POST'])
def predict():
    # Get the input text
    input_text = request.json['text']

    # Cleaned text
    cleaned_text = preprocess_text(input_text)

    # Tokenize the text
    input_sequence = tokenizer.texts_to_sequences([cleaned_text])
    maxlen = max([len(seq) for seq in input_sequence])
    X = pad_sequences(input_sequence, maxlen=maxlen)

    # Make prediction
    y_pred_lstm = model_lstm.predict(X)
    y_pred_lstm = np.argmax(y_pred_lstm)

    return jsonify({'y_pred_lstm': int(y_pred_lstm)})

@app.route('/predict_nn', methods=['POST'])
def predict_nn():
    # Get the input text
    input_text = request.json['text']

    # Preprocess the text
    cleaned_text = preprocess_text(input_text)

    # Tokenize the text
    input_sequence = tokenizer.texts_to_sequences([cleaned_text])
    maxlen = max([len(seq) for seq in input_sequence])
    X = pad_sequences(input_sequence, maxlen=maxlen)

    # Make prediction using Neural Network model
    y_pred_nn = model_nn.predict(X)
    y_pred_nn = np.argmax(y_pred_nn)

    return jsonify({'y_pred_nn': int(y_pred_nn)})

if __name__ == '__main__':
    app.run(debug=True)