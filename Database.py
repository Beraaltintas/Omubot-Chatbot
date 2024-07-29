import json
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Veritabanı bağlantısı
engine = create_engine('sqlite:///unknown_questions.db')
Base = declarative_base()

# Soru tablosu tanımı
class Question(Base):
    __tablename__ = 'questions'
    id = Column(Integer, primary_key=True)
    question = Column(Text, nullable=False, unique=True)
    answer = Column(Text, nullable=True)

Base.metadata.create_all(engine)

# Veritabanından soruları yükleme
def load_questions_from_database():
    Session = sessionmaker(bind=engine)
    session = Session()

    questions = session.query(Question).all()
    intents = []

    for question in questions:
        intent = {
            "tag": "custom_tag",  # Burada isteğe bağlı olarak etiket belirleyebilirsiniz.
            "patterns": [question.question],
            "responses": [question.answer if question.answer else ""],
            "context_set": ""
        }
        intents.append(intent)

    return {"intents": intents}

# JSON dosyasına soruları kaydetme
def save_questions_to_json(intents_data, filename='questions_from_database.json'):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(intents_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    intents_data = load_questions_from_database()
    save_questions_to_json(intents_data)
    print("Veritabanındaki sorular başarıyla JSON dosyasına aktarıldı.")
