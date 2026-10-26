import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

SYSTEM_PROMPT = """
You are Kintu AI, a friendly Ugandan AI assistant.
Use natural Luganda as your main language.
Use simple British English when helpful.
If the user writes in English, you may answer in English.
Be polite, helpful and patient.
Explain things simply.
"""

@app.route("/", methods=["GET"])
def home():
    return "Kintu AI backend is running!"

@app.route("/chat", methods=["POST"])
def chat():
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        return jsonify({"error": "GROQ_API_KEY is missing from Render"}), 500

    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"error": "Please enter a message"}), 400

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "openai/gpt-oss-20b",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": message}
                ],
                "max_completion_tokens": 700
            },
            timeout=60
        )

        if not response.ok:
            app.logger.error("Groq API returned HTTP %s", response.status_code)
            return jsonify({"error": "Groq could not answer. Check Render logs."}), 502

        result = response.json()
        reply = result["choices"][0]["message"]["content"]
        return jsonify({"reply": reply})

    except requests.RequestException:
        app.logger.exception("Groq connection failed")
        return jsonify({"error": "Connection problem. Please try again."}), 502

    except (KeyError, IndexError, TypeError):
        app.logger.exception("Unexpected Groq response")
        return jsonify({"error": "Unexpected AI response."}), 502

if __name__ == "__main__":
    app.run()
