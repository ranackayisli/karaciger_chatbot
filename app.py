from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
import google.generativeai as genai

app = Flask(__name__)

# --- Gemini ayarları ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# --- CSV yükleme ---
df = pd.read_csv("questions.csv")

# TF-IDF vektörleştirici ile soru gövdelerini hazırla
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["question"])

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    user_tfidf = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_tfidf, tfidf_matrix)

    # En benzer soruyu bul
    max_index = similarity.argmax()
    matched_question = df.iloc[max_index]["question"]
    matched_answer = df.iloc[max_index]["answer"]
    match_score = round(similarity[0][max_index] * 100, 1)

    # Eğer benzerlik %70’in altındaysa veya eşleşen soru içinde "karaciğer" yoksa → Gemini
    if match_score < 70 or "karaciğer" not in matched_question.lower():
        try:
            response = gemini_model.generate_content(
                f"Kullanıcı şunu sordu: {user_input}\n\n"
                f"Sen bir karaciğer sağlığı asistanısın. "
                f"Eğer soru alakasızsa 'Bu soru karaciğer sağlığı ile ilgili değil' diye cevap ver. "
                f"Kısa ve anlaşılır yanıt ver."
            )
            gemini_answer = response.text.strip() if response.text else "Cevap alınamadı."
            return jsonify({
                "matched_question": None,
                "matched_answer": gemini_answer,
                "similarity": match_score,
                "source": "gemini"
            })
        except Exception as e:
            return jsonify({
                "matched_question": None,
                "matched_answer": f"Gemini API hatası: {str(e)}",
                "similarity": match_score,
                "source": "gemini"
            })

    # Eğer %70 üzerindeyse CSV cevabı dön
    return jsonify({
        "matched_question": matched_question,
        "matched_answer": matched_answer,
        "similarity": match_score,
        "source": "faq"
    })

if __name__ == "__main__":
    app.run(debug=True, port=5001)  # 5000 dolu olabilir, garanti olsun diye 5001