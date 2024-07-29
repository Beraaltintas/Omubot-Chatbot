import requests
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
import pandas as pd
import random
from fuzzywuzzy import fuzz
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS

# Database setup
engine = create_engine('sqlite:///unknown_questions.db')
Base = declarative_base()

class Question(Base):
    __tablename__ = 'questions'
    id = Column(Integer, primary_key=True)
    question = Column(Text, nullable=False, unique=True)
    answer = Column(Text, nullable=True)

Base.metadata.drop_all(engine)  # Drop existing tables if necessary
Base.metadata.create_all(engine)  # Create tables with the updated schema

Session = sessionmaker(bind=engine)
session = Session()

def log_unknown_question(question):
    try:
        unknown_question = Question(question=question)
        session.add(unknown_question)
        session.commit()
    except IntegrityError:
        session.rollback()  # Roll back if the same question already exists

# Load JSON data and process it
def load_json_file(filename):
    with open(filename, encoding='utf-8') as f:
        file = json.load(f)
    return file

def extract_json_info(json_file):
    rows = []
    for intent in json_file['intents']:
        for pattern in intent['patterns']:
            for response in intent['responses']:
                rows.append({'Pattern': pattern, 'Tag': intent['tag'], 'Response': response})
    df = pd.DataFrame(rows)
    return df

# Load JSON file
json_file_path = 'C:\\Users\\mbera\\Desktop\\omubot\\Dataset.json'
intents = load_json_file(json_file_path)

# Create DataFrame
df = extract_json_info(intents)

# Map labels
labels = df['Tag'].unique().tolist()
id2label = {id: label for id, label in enumerate(labels)}
label2id = {label: id for id, label in enumerate(labels)}

def save_questions_to_json(df, filename='questions.json'):
    questions_data = df.to_dict(orient='records')
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(questions_data, f, ensure_ascii=False, indent=4)

save_questions_to_json(df)

def load_questions_from_json(filename='questions.json'):
    with open(filename, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    return pd.DataFrame(questions_data)

df_questions = load_questions_from_json()

# Model and tokenizer paths
model_path = "C:\\Users\\mbera\\Desktop\\omubot\\omubotonline"
tokenizer_path = model_path

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Flask app setup
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def get_daily_meal():
    url = 'https://sks.omu.edu.tr/gunun-yemegi/'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    meal_table = soup.find('table', class_='has-background')
    
    if meal_table:
        rows = meal_table.find_all('tr')
        meals_info = []
        for row in rows:
            cols = row.find_all('td')
            cols = [col.text.strip() for col in cols if col.text.strip()]
            if cols:
                meals_info.append(cols)
        
        meal_info = []
        for meal in meals_info[1:-1]:
            meal_info.append(f"{meal[0]}: {meal[1]}")
        
        return "\n".join(meal_info)
    
    return "Günün yemeği bilgisine ulaşılamıyor."

def get_current_date():
    today = datetime.today()
    return f"Bugün {today.strftime('%d')} {get_month_name(today.month)} {today.strftime('%Y')} {get_day_name(today.strftime('%A'))}"

def get_tomorrow_date():
    tomorrow = datetime.today() + timedelta(days=1)
    return f"Yarın {tomorrow.strftime('%d')} {get_month_name(tomorrow.month)} {tomorrow.strftime('%Y')} {get_day_name(tomorrow.strftime('%A'))}"

def get_current_time():
    now = datetime.now()
    return now.strftime("%H:%M:%S")

def get_month_name(month):
    month_names = ['Ocak', 'Şubat', 'Mart', 'Nisan', 'Mayıs', 'Haziran', 'Temmuz', 'Ağustos', 'Eylül', 'Ekim', 'Kasım', 'Aralık']
    return month_names[month - 1]

def get_day_name(day):
    day_names = {
        'Monday': 'Pazartesi',
        'Tuesday': 'Salı',
        'Wednesday': 'Çarşamba',
        'Thursday': 'Perşembe',
        'Friday': 'Cuma',
        'Saturday': 'Cumartesi',
        'Sunday': 'Pazar'
    }
    return day_names.get(day, day)



def predict(text):
    text_lower = text.lower()

    if any(keyword in text_lower for keyword in ["yemekte ne var", "bugün yemekte ne var", "bugünkü yemekler neler", "bugün menüde ne var"]):
        return get_daily_meal()

    if any(keyword in text_lower for keyword in ["tarih", "bugünün tarihi", "bugün hangi gün"]):
        return get_current_date()

    if any(keyword in text_lower for keyword in ["saat kaç", "saat nedir", "saat"]):
        return get_current_time()

    if any(keyword in text_lower for keyword in ["bugün günlerden ne", "bugün günlerden neyse"]):
        today = datetime.today()
        day = today.strftime("%d")
        month_names = ['Ocak', 'Şubat', 'Mart', 'Nisan', 'Mayıs', 'Haziran', 'Temmuz', 'Ağustos', 'Eylül', 'Ekim', 'Kasım', 'Aralık']
        month = month_names[int(today.strftime("%m")) - 1]
        year = today.strftime("%Y")
        day_name = get_day_name(today.strftime("%A"))
        return f"Bugün {day} {month} {year} {day_name}."

    if any(keyword in text_lower for keyword in ["yarın ayın kaçı", "yarın hangi gün", "yarın günlerden ne"]):
        return get_tomorrow_date()

    for pattern in df_questions['Pattern']:
        pattern_lower = pattern.lower()
        similarity_ratio = fuzz.token_sort_ratio(pattern_lower, text_lower)
        similarity_threshold = 55

        if similarity_ratio >= similarity_threshold:
            inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
            outputs = model(**inputs)
            probs = outputs.logits.softmax(dim=1)
            pred_label_idx = probs.argmax()
            pred_label = model.config.id2label[pred_label_idx.item()]

            if pred_label not in df['Tag'].values:
                log_unknown_question(text)
                return "Bu sorunuzu yanıtlayamıyorum, en kısa sürede yanıtlayacağım."

            responses = df[df['Tag'] == pred_label]['Response']

            if responses.empty:
                log_unknown_question(text)
                return "Bu sorunuzu yanıtlayamıyorum, en kısa sürede yanıtlayacağım."

            response = random.choice(responses.tolist())
            response = response.encode('utf-8').decode('utf-8', errors='ignore')

            return response

    log_unknown_question(text)
    return "Bu sorunuzu yanıtlayamıyorum, en kısa sürede yanıtlayacağım."


@app.route('/predict', methods=['POST'])
def get_prediction():
    data = request.get_json(force=True)
    question = data.get('question', '')
    answer = predict(question)
    return jsonify({'answer': answer})

if __name__ == "__main__":
   app.run(host='0.0.0.0', port=8000)
#if __name__ == "__main__":
#    app.run(host='192.168.2.109', port=8000)