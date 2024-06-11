# NEW
import pandas as pd
import re
import sqlite3
from num2words import num2words
from flask import Flask, jsonify
from flask import request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from

app = Flask(__name__)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\\t|\\n|\\u', ' ', text)
    text = re.sub(r"https?:[^\s]+", ' ', text)
    text = re.sub(r"[^\w\s+]", '', text)
    text = re.sub(r'rt|user', ' ', text)
    text = re.sub(r'[\\x]+[a-z0-9]{2}', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = text.replace('_', ' ')
    text = re.sub(r'(\d+)', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'(\d+)', lambda x: num2words(int(x.group(0)), lang='id'), text)
    return text

alay_df = pd.read_csv('new_kamusalay.csv', encoding='latin-1', header=None)
alay_filter = dict(zip(alay_df[0], alay_df[1]))

def normalisasi_alay(text):
    return ' '.join(alay_filter.get(word, word) for word in text.split(' '))

app.json_encoder = LazyJSONEncoder
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

@swag_from("docs/hello_world.yml", methods=['GET'])
@app.route('/', methods=['GET'])
def hello_world():
    json_response = {
        'status_code': 200,
        'description': "Menyapa Hello World",
        'data': "Hello World",
    }

    response_data = jsonify(json_response)
    return response_data

# @swag_from("C://Users/Admin/Documents/GitHub/231000001-19-pbk-cleansing-gold/docs/text.yml", methods=['GET'])
# @app.route('/text', methods=['GET'])
# def text():
#     json_response = {
#         'status_code': 200,
#         'description': "Original Teks",
#         'data': "Halo, apa kabar semua?",
#     }

#     response_data = jsonify(json_response)
#     return response_data

# @swag_from("C://Users/Admin/Documents/GitHub/231000001-19-pbk-cleansing-gold/docs/text_clean.yml", methods=['GET'])
# @app.route('/text-clean', methods=['GET'])
# def text_clean():
#     json_response = {
#         'status_code': 200,
#         'description': "Teks yang sudah dibersihkan",
#         'data': re.sub(r'[^a-zA-Z0-9]', ' ', "Halo, apa kabar semua?"),
#     }

#     response_data = jsonify(json_response)
#     return response_data

@swag_from("docs/text_processing.yml", methods=['POST'])
@app.route('/text-processing', methods=['POST'])
def text_processing():
    text = request.form.get('text')
    cleaned_text = clean_text(text)
    normalized_text = normalisasi_alay(cleaned_text)

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
    data = pd.read_csv('data.csv', encoding='latin-1')

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