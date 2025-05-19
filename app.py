from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
import os

app = Flask(__name__)

# 🔐 API Anahtarı
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or "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"

# 🧠 Embedding modeli
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# 📚 Soru-cevap veri kümesini yükle
df = pd.read_csv("soru_cevap_veri.csv", encoding="ISO-8859-9", sep=";", header=1)  # CSV dosyasını bu adla aynı klasöre koy
df = df.dropna()  # Eksik verileri temizle
questions = df["Soru"].tolist()
answers = df["Cevap"].tolist()
question_embeddings = embed_model.encode(questions, convert_to_tensor=True)

# 🔍 En yakın soruyu bul
def find_most_similar_answer(user_input, top_n=1):
    user_embedding = embed_model.encode([user_input], convert_to_tensor=True)
    user_embedding_np = user_embedding.cpu().numpy()
    question_embeddings_np = question_embeddings.cpu().numpy()
    similarities = cosine_similarity(user_embedding_np, question_embeddings_np)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    matched_qas = [(questions[i], answers[i]) for i in top_indices]
    return matched_qas[0][1]

# 🧠 Mixtral API isteği
def mixtral_response(prompt, context=""):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistralai/mixtral-8x7b-instruct",
        "messages": [
            {"role": "system", "content": "Sen bir Türk akademik danışman botsun. Kullanıcıya sadece düzgün ve doğal Türkçe cevap ver. İngilizce veya bozuk cümleler kurma."},
            {"role": "user", "content": f"Kullanıcının sorusu: {prompt}\n\nİlgili bilgi: {context}"}
        ]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=data, headers=headers)

    try:
        json_resp = response.json()
        return json_resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("❌ JSON Hatası veya API Hatası:", response.text)
        print("Exception:", e)
        return "Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin."

# 💬 HTML arayüz
HTML_PAGE = '''
<!doctype html>
<html>
  <head>
    <title>İzüBot (Mixtral + Veri Seti)</title>
    <style>
      body { font-family: Arial; background: #f0f2f5; padding: 20px; }
      .chatbox { background: white; border-radius: 10px; padding: 20px; max-width: 600px; margin: auto; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
      .msg { margin: 10px 0; }
      .user { text-align: right; }
      .bot { text-align: left; color: #0056b3; }
      input, button { padding: 10px; width: 80%; margin-top: 10px; border-radius: 5px; border: 1px solid #ccc; }
    </style>
  </head>
  <body>
    <div class="chatbox">
      <h2>İzüBot 🤖 (Mixtral + Veri Seti)</h2>
      <div id="chatlog"></div>
      <input type="text" id="userInput" placeholder="Bir şeyler yaz..." onkeypress="if(event.key==='Enter')sendMessage()" />
      <button onclick="sendMessage()">Gönder</button>
    </div>
    <script>
      async function sendMessage() {
        const input = document.getElementById("userInput");
        const message = input.value;
        if (!message.trim()) return;
        const chatlog = document.getElementById("chatlog");
        chatlog.innerHTML += '<div class="msg user"><strong>Sen:</strong> ' + message + '</div>';
        input.value = "";

        const response = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: message })
        });

        const data = await response.json();
        chatlog.innerHTML += '<div class="msg bot"><strong>İzüBot:</strong> ' + data.response + '</div>';
        chatlog.scrollTop = chatlog.scrollHeight;
      }
    </script>
  </body>
</html>
'''

@app.route("/")
def home():
    return render_template_string(HTML_PAGE)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    matched_answer = find_most_similar_answer(user_input)
    full_response = mixtral_response(user_input, context=matched_answer)
    return jsonify({"response": full_response})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port)
