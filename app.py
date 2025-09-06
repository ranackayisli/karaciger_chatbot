from flask import Flask, request, jsonify, render_template
import os
import unicodedata
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import google.generativeai as genai

# =======================
# Config
# =======================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in environment (.env).")

CSV_PATH = os.getenv("CSV_PATH", "questions.csv")
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.70"))  # Flask örneğinde 70 idi

# Zorunlu anahtar kelime(ler): sadece "karaciğer" istiyorsan bu seti tek elemanla bırak
REQUIRED_TOKENS = {"karaciger"}  # istersen {"karaciger", "liver", "hepatit", "siroz"} yap

# =======================
# Helpers
# =======================
def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def tr_simplify(s: str) -> str:
    """Türkçe diakritikleri sadeleştir (karaciğer -> karaciger), lower-case."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s.lower()

def contains_required_keyword(user_q: str) -> bool:
    t = tr_simplify(user_q)
    return any(tok in t for tok in REQUIRED_TOKENS)

# =======================
# Gemini
# =======================
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))

def ask_gemini(user_input: str) -> str:
    prompt = (
        "Sen dikkatli bir tıbbi asistansın. "
        "Sorulara TÜRKÇE yanıt ver. "
        "Kısa ve anlaşılır cevaplar yaz. "
        "Eğer soru karaciğer sağlığı ile ilgili değilse, "
        "'Bu soru karaciğer sağlığı ile ilgili değil.' diye cevap ver. "
        "Gerekli olduğunda standart tıbbi uyarıları ekle.\n\n"
        f"Kullanıcı sorusu:\n{user_input}\n"
    )
    resp = gemini_model.generate_content(prompt)
    return (getattr(resp, "text", "") or "").strip() or "Cevap alınamadı."

# =======================
# Data + Vectorizer
# =======================
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

for col in ("question", "answer"):
    if col not in df.columns:
        raise ValueError(f"CSV must contain '{col}' column.")

df["question_norm"] = df["question"].fillna("").map(normalize_text)
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
tfidf_matrix = vectorizer.fit_transform(df["question_norm"])

# =======================
# Flask
# =======================
app = Flask(__name__)

@app.route("/")
def home():
    # templates/chat.html varsa render_template ile; yoksa send_from_directory kullan
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(force=True)
    user_input = (payload or {}).get("message", "")
    user_norm = normalize_text(user_input)

    # 0) Zorunlu anahtar kelime filtresi: "karaciğer" geçmiyorsa cevaplama
    if not contains_required_keyword(user_input):
        return jsonify({
            "matched_question": None,
            "matched_answer": "",
            "similarity": 0.0,
            "source": "blocked"
        })

    # 1) Exact match kontrolü
    exact_mask = (df["question_norm"] == user_norm)
    if exact_mask.any():
        idx = int(exact_mask[exact_mask].index[0])
        return jsonify({
            "matched_question": df.iloc[idx]["question"],
            "matched_answer": df.iloc[idx]["answer"],
            "similarity": 100.0,
            "source": "faq"
        })

    # 2) TF-IDF benzerlik (question-only)
    user_vec = vectorizer.transform([user_norm])
    sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
    best_idx = int(sims.argmax())
    match_score = float(sims[best_idx] * 100.0)
    matched_q = df.iloc[best_idx]["question"]
    matched_a = df.iloc[best_idx]["answer"]

    # 3) Eşik kontrolü: altındaysa Gemini
    if match_score < SIM_THRESHOLD * 100:
        try:
            g_answer = ask_gemini(user_input)
            return jsonify({
                "matched_question": None,
                "matched_answer": g_answer,
                "similarity": round(match_score, 1),
                "source": "gemini"
            })
        except Exception as e:
            return jsonify({
                "matched_question": None,
                "matched_answer": f"Gemini API error: {e}",
                "similarity": round(match_score, 1),
                "source": "gemini"
            })

    # 4) Eşik üstünde → CSV cevabı
    return jsonify({
        "matched_question": matched_q,
        "matched_answer": matched_a,
        "similarity": round(match_score, 1),
        "source": "faq"
    })

if __name__ == "__main__":
    # 5000 doluysa değiştir
    app.run(debug=True, port=5001)