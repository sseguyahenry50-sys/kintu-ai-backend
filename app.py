import os
import uuid
import base64
import logging
import urllib.parse
from threading import Lock

import requests
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# CHANGED: 10MB for photos, was 1MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper()
)

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
- Be patient and respectful. Your boss is sseguya henry AKA (inkwire)
- Do not mention these instructions.

CULTURAL CONTEXT:
- Understand Uganda's everyday life, culture and local context.
- Use Ugandan examples when they help explain something.
- Do not assume every user lives in Kampala.
"""

GENERATED_IMAGES = {}
GENERATED_IMAGES_LOCK = Lock()

MAX_STORED_IMAGES = int(
    os.environ.get("MAX_STORED_IMAGES", "20")
)

def json_error(message, status_code):
    return jsonify({
        "success": False,
        "error": message
    }), status_code

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "success": True,
        "message": "Kintu AI backend is running!",
        "endpoints": [
            "/chat",
            "/generate-image",
            "/describe-image"
        ]
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "status": "ok"
    })

# ==========================================
# KINTU AI CHAT - GROQ (UNCHANGED)
# ==========================================

@app.route("/chat", methods=["POST"])
def chat():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        app.logger.error("GROQ_API_KEY is missing")
        return json_error("Chat service is not configured.", 500)
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        data = request.form
    message = data.get("message", "")
    if not isinstance(message, str):
        return json_error("Message must be text.", 400)
    message = message.strip()
    if not message:
        return json_error("Please enter a message.", 400)
    model = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
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
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": message}
                ],
                "max_completion_tokens": 900,
                "temperature": 0.4
            },
            timeout=(15, 60)
        )
        if not response.ok:
            app.logger.error("Groq error HTTP %s: %s", response.status_code, response.text[:1000])
            return json_error("Kintu AI could not answer. Please try again.", 502)
        result = response.json()
        reply = result["choices"][0]["message"]["content"]
        if not isinstance(reply, str) or not reply.strip():
            return json_error("Kintu AI returned an empty answer.", 502)
        return Response(reply.strip(), status=200, mimetype="text/plain; charset=utf-8")
    except requests.Timeout:
        app.logger.exception("Groq request timed out")
        return json_error("Kintu AI took too long to respond.", 504)
    except requests.RequestException:
        app.logger.exception("Groq connection failed")
        return json_error("Connection problem. Please try again.", 502)
    except (KeyError, IndexError, TypeError, ValueError):
        app.logger.exception("Unexpected Groq response")
        return json_error("Unexpected AI response.", 502)

# ==========================================
# IMAGE GENERATION - POLLINATIONS (UNCHANGED)
# ==========================================

@app.route("/generate-image", methods=["POST"])
def generate_image():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return json_error('Send JSON containing a "prompt" field.', 400)
    prompt = data.get("prompt", "")
    if not isinstance(prompt, str):
        return json_error("Prompt must be text.", 400)
    prompt = prompt.strip()
    if not prompt:
        return json_error("Please describe the image you want.", 400)
    if len(prompt) > 4000:
        return json_error("Your image description is too long.", 400)
    safe_prompt = urllib.parse.quote(prompt, safe='')
    seed = uuid.uuid4().hex[:8]
    quality = data.get("quality", "")
    if quality == "4K":
        width, height = 1024, 1024
    elif quality == "HD":
        width, height = 768, 768
    elif quality == "Full HD":
        width, height = 1024, 1024
    else:
        width, height = 1024, 1024
    image_url = (
        f"https://image.pollinations.ai/prompt/{safe_prompt}"
        f"?width={width}&height={height}&seed={seed}&nologo=true&model=flux"
    )
    app.logger.info(f"Generated image URL: {image_url}")
    return jsonify({
        "success": True,
        "image_url": image_url,
        "format": "png"
    }), 200

# ==========================================
# NEW: UNDERSTAND + DESCRIBE IMAGES - GEMINI
# ==========================================

@app.route("/describe-image", methods=["POST"])
def describe_image():
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        app.logger.error("GEMINI_API_KEY is missing")
        return json_error("Image understanding is not configured.", 500)

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return json_error('Send JSON with "image" and "prompt".', 400)

    # Kodular can send "prompt" or "message" - accept both
    prompt = data.get("prompt", "") or data.get("message", "") or "Describe this image in detail"
    if not isinstance(prompt, str):
        prompt = str(prompt)
    prompt = prompt.strip() or "Describe this image in detail"

    image_b64 = data.get("image", "") or data.get("image_base64", "")
    image_url = data.get("image_url", "")

    mime_type = "image/jpeg"
    image_bytes_b64 = ""

    # Option 1: Kodular sends base64
    if isinstance(image_b64, str) and len(image_b64) > 100:
        # Remove "data:image/jpeg;base64," if present
        if "," in image_b64 and "base64" in image_b64[:50]:
            parts = image_b64.split(",", 1)
            image_b64 = parts[1]
            # Try to detect mime from header
            if "png" in parts[0]:
                mime_type = "image/png"
            elif "webp" in parts[0]:
                mime_type = "image/webp"
        image_bytes_b64 = image_b64

    # Option 2: Kodular sends an image_url
    elif isinstance(image_url, str) and image_url.startswith("http"):
        try:
            img_resp = requests.get(image_url, timeout=20)
            if not img_resp.ok:
                return json_error("Could not download image from URL.", 400)
            # Guess mime
            content_type = img_resp.headers.get("Content-Type", "image/jpeg")
            if "png" in content_type:
                mime_type = "image/png"
            elif "webp" in content_type:
                mime_type = "image/webp"
            else:
                mime_type = "image/jpeg"
            image_bytes_b64 = base64.b64encode(img_resp.content).decode("utf-8")
        except requests.RequestException:
            return json_error("Could not download image from URL.", 502)

    if not image_bytes_b64:
        return json_error("Please send an image. Use 'image' as base64 or 'image_url'.", 400)

    # Send to Gemini - uses your SYSTEM_PROMPT so it answers in Luganda like Kintu
    try:
        gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
        gemini_payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": mime_type, "data": image_bytes_b64}}
                ]
            }],
            "generationConfig": {"temperature": 0.4, "maxOutputTokens": 900}
        }
        g_resp = requests.post(gemini_url, json=gemini_payload, timeout=(15, 60))
        if not g_resp.ok:
            app.logger.error("Gemini error %s: %s", g_resp.status_code, g_resp.text[:1000])
            return json_error("Kintu AI could not understand this image. Try again.", 502)

        result = g_resp.json()
        # Extract text
        description = ""
        try:
            candidates = result.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                description = "".join([p.get("text", "") for p in parts]).strip()
        except Exception:
            pass

        if not description:
            return json_error("Kintu AI returned an empty answer.", 502)

        return jsonify({
            "success": True,
            "description": description
        }), 200

    except requests.Timeout:
        return json_error("Kintu AI took too long to look at the image.", 504)
    except Exception:
        app.logger.exception("Describe image failed")
        return json_error("Could not understand image.", 502)

# ==========================================
# DISPLAY OLD GENERATED IMAGES
# ==========================================

@app.route("/generated-images/<image_id>.png", methods=["GET"])
def get_generated_image(image_id):
    with GENERATED_IMAGES_LOCK:
        image_bytes = GENERATED_IMAGES.get(image_id)
    if image_bytes is None:
        return Response("Image not found. It may have expired.", status=404, mimetype="text/plain")
    return Response(
        image_bytes,
        status=200,
        mimetype="image/png",
        headers={
            "Cache-Control": "public, max-age=3600",
            "X-Content-Type-Options": "nosniff"
        }
    )

@app.errorhandler(413)
def request_too_large(error):
    return json_error("Image is too large. Try a smaller photo.", 413)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
