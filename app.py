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

# 10MB for photos
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
# NEW: DESCRIBE IMAGE - GEMINI
# Supports 3 ways:
# 1. PostFile from Kodular (file upload) - NO EXTENSION NEEDED
# 2. JSON Base64 (old way)
# 3. JSON image_url
# ==========================================

@app.route("/describe-image", methods=["POST"])
def describe_image():
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        app.logger.error("GEMINI_API_KEY is missing")
        return json_error("Image understanding is not configured.", 500)

    # --- Get prompt from ANY place Kodular might send it ---
    # URL ?prompt=..., form, or JSON
    prompt = ""
    # 1. URL query: /describe-image?prompt=What is this?
    prompt = request.args.get("prompt", "") or ""
    # 2. Form field (when using PostFile)
    if not prompt:
        prompt = request.form.get("prompt", "") or request.form.get("message", "") or ""
    # 3. JSON body
    data = request.get_json(silent=True)
    if not prompt and isinstance(data, dict):
        prompt = data.get("prompt", "") or data.get("message", "") or ""
    if not isinstance(prompt, str):
        prompt = str(prompt)
    prompt = prompt.strip() or "Describe this image in detail"

    mime_type = "image/jpeg"
    image_bytes = None

    # --- WAY 1: FILE UPLOAD from Kodular PostFile ---
    # Kodular PostFile sends file under key "image" or as any file
    if request.files:
        # Try common keys first
        file_obj = request.files.get("image")
        if file_obj is None and len(request.files) > 0:
            # Take first file whatever its name is
            file_obj = list(request.files.values())[0]
        if file_obj is not None and file_obj.filename:
            try:
                image_bytes = file_obj.read()
                fname = file_obj.filename.lower()
                if fname.endswith(".png"):
                    mime_type = "image/png"
                elif fname.endswith(".webp"):
                    mime_type = "image/webp"
                elif fname.endswith(".jpg") or fname.endswith(".jpeg"):
                    mime_type = "image/jpeg"
            except Exception:
                app.logger.exception("Failed to read uploaded file")

    # --- WAY 2: JSON Base64 ---
    if image_bytes is None and isinstance(data, dict):
        image_b64 = data.get("image", "") or data.get("image_base64", "")
        if isinstance(image_b64, str) and len(image_b64) > 100:
            if "," in image_b64 and "base64" in image_b64[:50]:
                parts = image_b64.split(",", 1)
                image_b64 = parts[1]
                if "png" in parts[0]:
                    mime_type = "image/png"
                elif "webp" in parts[0]:
                    mime_type = "image/webp"
            try:
                image_bytes = base64.b64decode(image_b64)
            except Exception:
                pass

    # --- WAY 3: JSON image_url ---
    if image_bytes is None and isinstance(data, dict):
        image_url = data.get("image_url", "") or request.args.get("image_url", "")
        if isinstance(image_url, str) and image_url.startswith("http"):
            try:
                img_resp = requests.get(image_url, timeout=20)
                if img_resp.ok:
                    image_bytes = img_resp.content
                    ct = img_resp.headers.get("Content-Type", "image/jpeg")
                    if "png" in ct:
                        mime_type = "image/png"
                    elif "webp" in ct:
                        mime_type = "image/webp"
            except requests.RequestException:
                pass

    if not image_bytes:
        return json_error("Please send an image. Use PostFile, or JSON with 'image' base64.", 400)

    image_bytes_b64 = base64.b64encode(image_bytes).decode("utf-8")

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
