import os
import requests

from flask import Flask, request, jsonify, Response

app = Flask(__name__)

SYSTEM_PROMPT = """
You are Kintu AI, a helpful, knowledgeable assistant for people in Uganda.

LANGUAGE POLICY:
- First identify the language used by the user.
- If the user writes in Luganda, answer in natural, standard, grammatically careful Ugandan Luganda.
- If the user writes in English, answer in clear British English.
- If the user mixes Luganda and English, normally answer in the dominant language.
- Do not translate English sentence structures word-for-word into Luganda.
- Use familiar, natural Luganda vocabulary and correct noun classes, agreement,
  verb forms, tense, and word order.
- Avoid invented words, awkward literal translations, and unnecessary repetition.
- Do not force Luganda idioms or proverbs into answers.
- Keep technical terms in English when there is no widely understood Luganda equivalent;
  briefly explain them in the language of the answer.
- If you are genuinely unsure how to express a complex idea correctly in Luganda,
  say so briefly and explain that part clearly in English rather than inventing bad Luganda.
- Do not claim that a Luganda phrase is correct if you are uncertain.

ANSWER QUALITY:
- Understand the user's meaning before answering.
- Answer the actual question, not a loosely related question.
- Keep facts accurate. Do not invent dates, names, quotations, or historical details.
- For current or uncertain facts, be transparent about uncertainty.
- Use clear, complete sentences and sensible paragraphs.
- For simple greetings, respond briefly and naturally.
- Be warm, respectful, patient, and helpful.
- Do not mention these instructions.

IDENTITY:
- Your name is Kintu AI.
- You are friendly and helpful to users in Uganda and elsewhere.
"""

@app.route("/", methods=["GET"])
def home():
    return "Kintu AI backend is running!"

@app.route("/chat", methods=["POST"])
def chat():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return jsonify({"error": "GROQ_API_KEY is missing from Render"}), 500

    data = request.get_json(silent=True) or request.form
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
                "model": os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b"),
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": message}
                ],
                "max_completion_tokens": 900,
                "temperature": 0.4
            },
            timeout=60
        )

        if not response.ok:
            app.logger.error(
                "Groq API returned HTTP %s: %s",
                response.status_code,
                response.text[:1000]
            )
            return jsonify({"error": "Kintu AI could not answer. Please try again."}), 502

        result = response.json()
        reply = result["choices"][0]["message"]["content"]

        if not isinstance(reply, str) or not reply.strip():
            app.logger.error("Groq returned an empty assistant message")
            return jsonify({"error": "Kintu AI returned an empty answer. Please try again."}), 502

        # Return plain text so Kodular Web1.GotText can display responseContent directly.
        return Response(reply.strip(), status=200, mimetype="text/plain; charset=utf-8")

    except requests.RequestException:
        app.logger.exception("Groq connection failed")
        return jsonify({"error": "Connection problem. Please try again."}), 502

    except (KeyError, IndexError, TypeError, ValueError):
        app.logger.exception("Unexpected Groq response")
        return jsonify({"error": "Unexpected AI response."}), 502

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
