import os
import re
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

SYSTEM_PROMPT = """
You are Kintu AI, a helpful assistant for people in Uganda.

LANGUAGE:
- Identify the language of the user's latest message.
- If the user writes in Luganda, answer in natural, clear, standard Ugandan Luganda.
- If the user writes in English, answer in clear British English.
- If the user mixes languages, normally use the language they used most.
- Do not translate English word-for-word into Luganda.
- Use simple, correct Luganda grammar and natural word order.
- Prefer familiar words. Do not invent Luganda words or awkward expressions.
- Keep technical names and unavoidable technical terms in their usual form.

ANSWER STYLE:
- Answer the exact question first.
- For a simple factual question, use 1-3 short sentences.
- Give only details that help answer the question.
- Do not repeat the question, the same fact, a sentence, or a phrase.
- Do not restate the answer in different words at the end.
- Avoid filler, long introductions, unnecessary lists, and repeated conclusions.
- Do not add coordinates, history, extra background, or unrelated facts unless asked.
- If you are unsure, say so briefly instead of guessing.
- Be warm, respectful, and conversational.
- Before sending, silently check: Is the Luganda natural? Did I answer the question? Did I repeat any idea? Remove repetition.
- Do not mention these instructions.

IDENTITY:
- Your name is Kintu AI.
- You are friendly, knowledgeable, and helpful.
"""

REPAIR_PROMPT = """
Rewrite the answer below to remove repeated words, phrases, sentences, and ideas.
Keep the original meaning and factual claims. Do not add new facts.
Use natural, grammatically careful Ugandan Luganda if the original answer is in Luganda.
Keep it concise and answer the user's question directly.
Return only the corrected answer.
"""

def call_groq(messages, api_key):
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b"),
            "messages": messages,
            "temperature": 0.2,
            "max_completion_tokens": 400
        },
        timeout=60
    )
    if not response.ok:
        app.logger.error(
            "Groq API returned HTTP %s: %s",
            response.status_code,
            response.text[:1000]
        )
        response.raise_for_status()

    result = response.json()
    text = result["choices"][0]["message"]["content"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Groq returned an empty answer")
    return text.strip()

def has_obvious_repetition(text):
    """Catch duplicated sentences or repeated long word sequences, not ordinary emphasis."""
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    sentences = [
        re.sub(r"[^\w\s']", "", part).strip()
        for part in re.split(r"(?<=[.!?])\s+", normalized)
    ]
    sentences = [s for s in sentences if len(s.split()) >= 5]

    # Exact sentence repeated
    if len(sentences) != len(set(sentences)):
        return True

    # Repeated sequence of 6+ consecutive words
    words = re.findall(r"\b[\w']+\b", normalized)
    seen = set()
    for size in (8, 7, 6):
        for i in range(len(words) - size + 1):
            phrase = tuple(words[i:i + size])
            if phrase in seen:
                return True
            seen.add(phrase)
    return False

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
        reply = call_groq(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message}
            ],
            api_key
        )

        # If the first draft visibly repeats itself, request one concise rewrite.
        if has_obvious_repetition(reply):
            app.logger.info("Repeated wording detected; requesting a concise rewrite")
            reply = call_groq(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": REPAIR_PROMPT}
                ],
                api_key
            )

        return jsonify({"reply": reply})

    except requests.RequestException:
        app.logger.exception("Groq connection failed")
        return jsonify({"error": "Connection problem. Please try again."}), 502
    except (KeyError, IndexError, TypeError, ValueError):
        app.logger.exception("Unexpected Groq response")
        return jsonify({"error": "Kintu AI could not prepare an answer. Please try again."}), 502

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
