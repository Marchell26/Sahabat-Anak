import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
import json
import random
from flask import Flask, render_template, request, jsonify
from difflib import SequenceMatcher
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


print("Files in current dir:", os.listdir(os.getcwd()))
# === Load model & data ===
model = load_model(os.path.join(BASE_DIR, 'model.h5'))
intents = json.load(open(os.path.join(BASE_DIR, 'data.json'), encoding="utf-8"))
words = pickle.load(open(os.path.join(BASE_DIR, 'texts.pkl'), 'rb'))
classes = pickle.load(open(os.path.join(BASE_DIR, 'labels.pkl'), 'rb'))

# load tokenizer
with open(os.path.join(BASE_DIR, 'tokenizer.pkl'), 'rb') as handle:
    tokenizer = pickle.load(handle)

# dapatkan max_len dari training
max_len = model.input_shape[1]

# ===== Stopwords Indonesia + custom =====
stop_words = set(stopwords.words("indonesian"))
extra_stops = {"apakah", "apanya", "bagaimana", "cara", "upaya", "agar",
               "mengapa", "kenapa", "saja", "aja", "merupakan", "ialah",
               "adalah", "untuk", "yang", "itu", "apa", "manfaat", "besi"}
stop_words = stop_words.union(extra_stops)

# ===== NLP helpers =====
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return " ".join(sentence_words)

def seq(sentence):
    seqs = tokenizer.texts_to_sequences([sentence])
    return pad_sequences(seqs, maxlen=max_len, padding='post')

def is_noise(sentence):
    import re
    s = sentence.strip()
    if len(s) < 3:
        return True
    if not re.search(r'[a-zA-Z]', s):
        return True
    alnum = re.sub(r'[^a-zA-Z0-9]', '', s)
    if len(alnum) / len(s) < 0.3:
        return True
    return False

# === Similarity helper ===
def sentence_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# === FIXED predict_class ===
def predict_class(sentence, model):
    try:
        clean_sentence = clean_up_sentence(sentence)
        seq_input = seq(clean_sentence)
        res = model.predict(seq_input)[0]
    except Exception:
        return []

    print("DEBUG >>", dict(zip(classes, [round(float(r), 3) for r in res])))

    tag_idx = np.argmax(res)
    confidence = float(res[tag_idx])

    THRESHOLD = 0.9
    if confidence < THRESHOLD:
        print("DEBUG >> confidence rendah, fallback")
        return []

    predicted_intent = classes[tag_idx]

    # === NEW: cek similarity + keyword coverage (dengan stopwords) ===
    for intent in intents["intents"]:
        if intent["tag"] == predicted_intent:
            patterns = intent.get("patterns", [])
            if not patterns:
                return []

            sims = [sentence_similarity(clean_sentence, p.lower()) for p in patterns]
            max_sim = max(sims) if sims else 0

            # keywords (filter stopwords)
            keywords = set([w for p in patterns for w in p.lower().split() if w not in stop_words])
            words_in_input = set([w for w in clean_sentence.split() if w not in stop_words])

            overlap = keywords.intersection(words_in_input)

            print(f"DEBUG >> intent={predicted_intent}, max_sim={max_sim}, overlap={overlap}")

            # lebih ketat: butuh similarity cukup tinggi DAN overlap keyword
            if max_sim < 0.75 or not overlap:
                print("DEBUG >> similarity rendah atau tidak ada overlap keyword, fallback")
                return []

    if predicted_intent == "fallback":
        return []

    return [{"intent": predicted_intent, "probability": str(confidence)}]

# === Ambil 1 response mentah (bisa string/dict) sesuai intent ===
def getResponse(ints, intents_json):
    if not ints:
        for intent in intents_json['intents']:
            if intent['tag'] == "fallback":
                # kembalikan apa adanya (string/dict)
                return random.choice(intent['responses'])
        return "Maaf, saya kurang paham dengan pertanyaan anda."
    tag = ints[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            # responses di dataset bisa string / {intro,points} / {sections}
            resp_list = i.get('responses', ["Maaf, saya belum punya jawaban."])
            return random.choice(resp_list)
    return "Maaf, saya belum punya jawaban untuk itu."

# ===== Recommendation builder =====
def build_recommendations(user_msg, ints, intents_json, total=6):
    recs = []
    if ints:
        tag = ints[0]['intent']
        for intent in intents_json['intents']:
            if intent.get('tag') == tag:
                patterns = intent.get('patterns', [])
                filtered = [p for p in patterns if p and p.strip() and p.lower() != (user_msg or "").lower()]
                if filtered:
                    take = min(3, len(filtered))
                    recs.extend(random.sample(filtered, take) if len(filtered) > take else filtered.copy())
                break
    all_patterns = []
    for intent in intents_json['intents']:
        all_patterns.extend(intent.get('patterns', []))
    remaining = [p for p in all_patterns if p and p.strip() and p.lower() != (user_msg or "").lower() and p not in recs]
    random.shuffle(remaining)
    need = total - len(recs)
    if need > 0 and remaining:
        recs.extend(remaining[:min(need, len(remaining))])
    return recs[:total]

# ===== Flask app =====
app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/index.html")
def index():
    return render_template("index.html")

@app.route("/tentang.html")
def tentang():
    return render_template("tentang.html")

@app.route("/gizi.html")
def gizi():
    return render_template("gizi.html")

@app.route("/pola.html")
def pola():
    return render_template("pola.html")

@app.route("/chatbot.html")
def chatbot():
    return render_template("chatbot.html")

@app.route("/edukasi.html")
def edukasi():
    return render_template("edukasi.html")

# === endpoint utama: jawab + rekomendasi 6 ===
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg', '').strip()

    if is_noise(userText):
        return jsonify({
            "response": ["Maaf, saya kurang paham dengan pertanyaan anda."],
            "recommendations": []
        })

    # --- 1) Pisahkan pertanyaan berdasarkan kata hubung yang umum ---
    delimiters = [" dan ", " serta ", " lalu ", " kemudian ", "?"]
    parts = [userText]
    for d in delimiters:
        temp = []
        for p in parts:
            subparts = [sp.strip() for sp in p.split(d) if sp.strip()]
            if len(subparts) > 1:
                temp.extend(subparts)
            else:
                temp.append(p)
        parts = temp

    responses_list = []

    if len(parts) == 1:
        ints = predict_class(userText, model)
        res = getResponse(ints, intents)  # string atau dict
        responses_list.append(res)
        recs = build_recommendations(userText, ints, intents, total=6)
        return jsonify({"response": responses_list, "recommendations": recs})

    # Lebih dari satu pertanyaan → proses tiap bagiannya
    for question_part in parts:
        ints_part = predict_class(question_part, model)
        res_part = getResponse(ints_part, intents)  # string atau dict
        responses_list.append(res_part)

    # Rekomendasi dari bagian pertama
    first_part = parts[0]
    ints_first = predict_class(first_part, model)
    recs = build_recommendations(first_part, ints_first, intents, total=6)

    return jsonify({"response": responses_list, "recommendations": recs})

# === endpoint untuk initial random questions ===
@app.route("/get_questions")
def get_questions():
    with open("data.json", "r", encoding="utf-8") as f:
        intents_data = json.load(f)
    questions = []
    for intent in intents_data.get("intents", []):
        questions.extend(intent.get("patterns", []))
    if len(questions) > 3:
        questions = random.sample(questions, 3)
    return jsonify(questions)

if __name__ == "__main__":
    app.run(debug=True)

