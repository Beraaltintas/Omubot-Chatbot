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

# Veritabanı bağlantısı
engine = create_engine('sqlite:///unknown_questions.db')
Base = declarative_base()

# Soru tablosu
class Question(Base):
    __tablename__ = 'questions'
    id = Column(Integer, primary_key=True)
    question = Column(Text, nullable=False, unique=True)
    answer = Column(Text, nullable=True)

Base.metadata.drop_all(engine)  # Mevcut tabloları sil (gerekirse)
Base.metadata.create_all(engine)  # Güncellenmiş şema ile tabloları oluştur

Session = sessionmaker(bind=engine)
session = Session()

def log_unknown_question(question):
    try:
        unknown_question = Question(question=question)
        session.add(unknown_question)
        session.commit()
    except IntegrityError:
        session.rollback()  # Eğer aynı soru zaten varsa işlemi geri al

# JSON dosyasından veriyi yükleme ve işleme
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

# JSON dosyasını yükle
json_file_path = 'C:\\Users\\mbera\\Desktop\\omubot\\Dataset.json'  # JSON dosyasının yolu
intents = load_json_file(json_file_path)

# Veri çerçevesi oluşturma
df = extract_json_info(intents)

# Etiketleri mapleme
labels = df['Tag'].unique().tolist()
id2label = {id: label for id, label in enumerate(labels)}
label2id = {label: id for id, label in enumerate(labels)}

# Veriyi JSON formatına dönüştürme ve kaydetme
def save_questions_to_json(df, filename='questions.json'):
    questions_data = df.to_dict(orient='records')
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(questions_data, f, ensure_ascii=False, indent=4)

# df'den soruları JSON formatına dönüştürüp kaydetme
save_questions_to_json(df)

# JSON'dan soruları yükleme
def load_questions_from_json(filename='questions.json'):
    with open(filename, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    return pd.DataFrame(questions_data)

# JSON'dan soruları yükleme
df_questions = load_questions_from_json()

# Model ve tokenizer yolları
model_path = ".\omubotonline"
tokenizer_path = model_path

# Tokenizer'ı ve modeli yükle
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Tahmin fonksiyonu güncellenmiş hali
def predict(text):
    # Fonksiyon çağrıldığında gelen soruyu küçük harfe dönüştür
    text_lower = text.lower()

    # Günün yemeği sorgusu varsa
    if any(keyword in text_lower for keyword in ["yemekte ne var", "bugün yemekte ne var", "bugünkü yemekler neler", "bugün menüde ne var", "yemek", "menü"]):
        return get_daily_meal()

    # Tarih sorgusu varsa
    if any(keyword in text_lower for keyword in ["tarih", "bugünün tarihi", "bugün hangi gün"]):
        return get_current_date()

    # Saat sorgusu varsa
    if any(keyword in text_lower for keyword in ["saat kaç", "saat nedir", "saat"]):
        return get_current_time()

    # "Bugün günlerden ne" gibi sorulara doğru cevap ver
    if any(keyword in text_lower for keyword in ["bugün günlerden ne", "bugün günlerden neyse"]):
        today = datetime.today()
        day = today.strftime("%d")
        month_names = ['Ocak', 'Şubat', 'Mart', 'Nisan', 'Mayıs', 'Haziran', 'Temmuz', 'Ağustos', 'Eylül', 'Ekim', 'Kasım', 'Aralık']
        month = month_names[int(today.strftime("%m")) - 1]
        year = today.strftime("%Y")
        day_name = get_day_name(today.strftime("%A"))
        return f"Bugün {day} {month} {year} {day_name}."

    # "Yarın ayın kaçı" gibi sorulara doğru cevap ver
    if any(keyword in text_lower for keyword in ["yarın ayın kaçı", "yarın hangi gün", "yarın günlerden ne"]):
        return get_tomorrow_date()

    # Diğer sorular için veritabanındaki desenler üzerinde döngü
    for pattern in df_questions['Pattern']:
        pattern_lower = pattern.lower()  # Veritabanındaki deseni küçük harfe dönüştür

        # Soru ve desen arasındaki benzerliği kontrol et
        similarity_ratio = fuzz.token_sort_ratio(pattern_lower, text_lower)

        # Benzerlik oranı eşiği
        similarity_threshold = 50  # %50 benzerlik eşiği

        if similarity_ratio >= similarity_threshold:
            # Eğer yeterince benzerlik varsa, modeli kullanarak tahminde bulun
            inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
            outputs = model(**inputs)
            probs = outputs.logits.softmax(dim=1)
            pred_label_idx = probs.argmax()
            pred_label = model.config.id2label[pred_label_idx.item()]

            # Etiketlerin DataFrame'de olup olmadığını kontrol et
            if pred_label not in df['Tag'].values:
                log_unknown_question(text)  # Bilinmeyen soruyu kaydet
                return "Bu sorunuzu yanıtlayamıyorum, en kısa sürede yanıtlayacağım."

            # Yanıtları DataFrame'den çek
            responses = df[df['Tag'] == pred_label]['Response']

            # Yanıtların boş olup olmadığını kontrol et
            if responses.empty:
                log_unknown_question(text)  # Bilinmeyen soruyu kaydet
                return "Bu sorunuzu yanıtlayamıyorum, en kısa sürede yanıtlayacağım."

            # Yanıtları doğru şekilde seç ve UTF-8'e dönüştür
            response = random.choice(responses.tolist())

            # UTF-8 encoding ve decoding işlemleri
            response = response.encode('utf-8').decode('utf-8', errors='ignore')

            return response

    # Eğer hiçbir desenle yeterince benzerlik bulunamazsa, soruyu kaydet ve placeholder cevap dön
    log_unknown_question(text)  # Bilinmeyen soruyu kaydet
    return "Bu sorunuzu yanıtlayamıyorum, en kısa sürede yanıtlayacağım."

def get_daily_meal():
    url = 'https://sks.omu.edu.tr/gunun-yemegi/'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Günün yemeği bilgilerini içeren tabloyu bulun
    meal_table = soup.find('table', class_='has-background')
    
    if meal_table:
        rows = meal_table.find_all('tr')
        meals_info = []
        for row in rows:
            cols = row.find_all('td')
            cols = [col.text.strip() for col in cols if col.text.strip()]
            if cols:
                meals_info.append(cols)
        
        # İlk sıra tarih, son sıra toplam kalori olduğu için onları çıkartıyoruz
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
    return day_names.get(day, day)  # Eğer gün ismi veritabanında yoksa kendisiyle geri döndür

if __name__ == "__main__":
    while True:
        text = input("Soru: ")
        if text.lower() in ['exit', 'quit', 'q']:
            break
        response = predict(text)
        print(f"Cevap: {response}")
