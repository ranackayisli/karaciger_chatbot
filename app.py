from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Soruları ve cevapları yükle
df = pd.read_csv("questions.csv", sep=",")

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

    return jsonify({
        "matched_question": matched_question,
        "matched_answer": matched_answer,
        "similarity": match_score
    })

if __name__ == "__main__":
    app.run(debug=True)