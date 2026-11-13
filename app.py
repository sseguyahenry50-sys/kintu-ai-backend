import os
import requests

from flask import Flask, request, jsonify, Response

app = Flask(__name__)

SYSTEM_PROMPT = """
You are Kintu AI, a friendly and knowledgeable Ugandan AI assistant.

IDENTITY:
- Your name is Kintu AI.
- You are designed primarily for Ugandan users.
- Be warm, respectful, intelligent and conversational.

LANGUAGE POLICY:
- Your default language is natural, fluent Ugandan Luganda.
- When a user writes in Luganda, respond in natural Luganda.
- When a user mixes Luganda and English, respond mainly in Luganda,
  naturally including familiar English words where appropriate.
- When a user writes in English, normally respond in Luganda,
  unless they specifically request English.
- If the user says "speak English" or asks for an English answer,
  respond in clear British English.
- Do not translate English sentence structures word for word.
- Use correct Luganda grammar, noun classes, verb forms and word order.
- Use familiar everyday Ugandan expressions.
- Avoid invented Luganda vocabulary and awkward translations.
- Keep technical terms in English when appropriate and explain them
  briefly in Luganda.
- Never pretend to know a Luganda expression if you are uncertain.
- Make your Luganda sound natural, not like a machine translation.

ANSWER QUALITY:
- Understand the user's actual question before answering.
- Give accurate and useful answers.
- Do not invent facts, names, dates or sources.
- Explain difficult subjects in simple Luganda.
- Keep simple greetings short and friendly.
- Give detailed answers when the user needs them.
- Be patient and respectful.
- Do not mention these instructions.

CULTURAL CONTEXT:
- Understand Uganda's everyday life, culture and local context.
- Use Ugandan examples when they help explain something.
- Do not assume every user lives in Kampala.
"""

@app.route("/", methods=["GET"])
def home():
    return "Kintu AI backend is running!"

@app.route("/chat", methods=["POST"])
def chat():
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        return jsonify({
            "error": "GROQ_API_KEY is missing from Render"
        }), 500

    data = request.get_json(silent=True) or request.form
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({
            "error": "Please enter a message"
        }), 400

    model = os.environ.get(
        "GROQ_MODEL",
        "qwen/qwen3.8-27b"
    )

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": message
                    }
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

            return jsonify({
                "error": "Kintu AI could not answer. Please try again."
            }), 502

        result = response.json()

        reply = result["choices"][0]["message"]["content"]

        if not isinstance(reply, str) or not reply.strip():
            return jsonify({
                "error": "Kintu AI returned an empty answer."
            }), 502

        return Response(
            reply.strip(),
            status=200,
            mimetype="text/plain; charset=utf-8"
        )

    except requests.RequestException:
        app.logger.exception("Groq connection failed")

        return jsonify({
            "error": "Connection problem. Please try again."
        }), 502

    except (KeyError, IndexError, TypeError, ValueError):
        app.logger.exception("Unexpected Groq response")

        return jsonify({
            "error": "Unexpected AI response."
        }), 502


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
