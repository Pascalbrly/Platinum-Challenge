# NEW
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import re
import nltk
import sqlite3
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from flask import Flask, jsonify
from flask import request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from

# Import JSON provider from flask
from flask.json.provider import DefaultJSONProvider

app = Flask(__name__)

class CustomJSONProvider(DefaultJSONProvider):
    def dumps(self, obj, **kwargs):
        return super().dumps(obj, cls=LazyJSONEncoder, **kwargs)

app.json = CustomJSONProvider(app)

def clean_text(text):
    text = re.sub(r'\\t|\\n|\\u', ' ', text) #Menghapus karakter khusus seperti tab, baris baru, karakter Unicode, dan backslash.
    text = re.sub(r"https?:[^\s]+", ' ', text)  # Menghapus http / https
    text = re.sub(r'(\b\w+)-\1\b', r'\1', text)
    text = re.sub(r'[\\x]+[a-z0-9]{2}', '', text)  # Menghapus karakter yang dimulai dengan '\x' diikuti oleh dua karakter huruf atau angka
    # text = re.sub(r'(\d+)', r' \1 ', text)  # Memisahkan angka dari teks
    text = re.sub(r'[^a-zA-Z]+', ' ', text)  # Menghapus karakter kecuali huruf, dan spasi
    text = re.sub(r'\brt\b|\buser\b', ' ', text) # Menghapus kata-kata 'rt' dan 'user'
    return text

alay_df = pd.read_csv('new_kamusalay.csv', encoding='latin-1', header=None)
alay_filter = dict(zip(alay_df[0], alay_df[1]))

def normalisasi_alay(text):
    return ' '.join(alay_filter.get(word, word) for word in text.split(' '))

factory = StemmerFactory()
stemer = factory.create_stemmer()

list_stopwords = set(stopwords.words('indonesian'))

# Tokenizing
def tokenize(text):
    return word_tokenize(text)

# Removing stopwords
def remove_stopwords(text):
    return [word for word in text if not word in list_stopwords]

# Stemming
def stemming(text):
    return [stemer.stem(word) for word in text]

# Convert list of words to a sentence
def words_to_sentence(list_words):
    return ' '.join(list_words)

swagger_template = dict(
info = {
    'title': 'Pascal Gold Challenge',
    'version': '1.0.0',
    'description': 'Dokumentasi API untuk Data Processing dan Modeling',
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

@swag_from("docs/text_processing.yml", methods=['POST'])
@app.route('/text-processing', methods=['POST'])
def text_processing():
    text = request.form.get('text')
    cleaned_text = clean_text(text)
    normalized_text = normalisasi_alay(cleaned_text)
    list_words = tokenize(normalized_text)
    list_words = remove_stopwords(list_words)
    stemmed_text = stemming(list_words)
    sentence = words_to_sentence(stemmed_text)

    conn = sqlite3.connect('binar_gold.db')
    cursor = conn.cursor()

    # Membuat tabel cleaned_text jika belum ada
    cursor.execute('create table if not exists insert_text(id integer primary key autoincrement, text text)')
    
    # Memasukkan data ke dalam tabel
    cursor.execute('insert into insert_text(text) values (?)', (normalized_text,))
    
    conn.commit()
    conn.close()

    json_response = {
        'status_code': 200,
        'description': "Teks yang sudah diproses",
        'data': normalized_text,
    }

    response_data = jsonify(json_response)
    return response_data


@swag_from("docs/text_processing_file.yml", methods=['POST'])
@app.route('/text-processing-file', methods=['POST'])
def text_processing_file():
    # Upladed file
    file = request.files.getlist('file')[0]

    # Import file csv ke Pandas
    data = pd.read_csv(file, encoding='latin-1')

    # Ambil teks yang akan diproses dalam format list
    texts = data.Tweet.to_list()

    # Lakukan cleansing pada teks
    cleaned_text = []
    for text in texts:
        text = clean_text(text)
        text = normalisasi_alay(text)
        cleaned_text.append(text)

    conn = sqlite3.connect('binar_gold.db')
    cursor = conn.cursor()

    # Membuat tabel cleaned_text
    cursor.execute('create table if not exists cleaned_text(id integer primary key autoincrement, text text)')

    # Insert data di tabel cleaned_text
    for text in cleaned_text:
        cursor.execute('insert into cleaned_text(text) values (?)', (text,))

    conn.commit()
    conn.close()

    json_response = {
        'status_code': 200,
        'description': "Teks yang sudah diproses",
        'data': cleaned_text,
    }

    response_data = jsonify(json_response)
    return response_data


if __name__ == '__main__':
   app.run()
