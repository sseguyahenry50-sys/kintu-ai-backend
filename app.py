
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

# Maximum photo upload size: 10 MB
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
- Make your Luganda sound natural, not like machine translation.

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

GENERATED_IMAGES = {}
GENERATED_IMAGES_LOCK = Lock()


def json_error(message, status_code):
    return jsonify({
        "success": False,
        "error": message
    }), status_code


def detect_image_mime(image_bytes, supplied_mime=""):
    """Detect the actual image type from its file signature."""

    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"

    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"

    if (
        image_bytes.startswith(b"RIFF")
        and image_bytes[8:12] == b"WEBP"
    ):
        return "image/webp"

    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"

    supplied_mime = (
        (supplied_mime or "")
        .split(";", 1)[0]
        .strip()
        .lower()
    )

    if supplied_mime in (
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/gif"
    ):
        return supplied_mime

    return ""


# ==========================================
# HOME AND HEALTH
# ==========================================

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
# KINTU AI CHAT - GROQ
# ==========================================

@app.route("/chat", methods=["POST"])
def chat():

    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        app.logger.error("GROQ_API_KEY is missing")
        return json_error(
            "Chat service is not configured.",
            500
        )

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        data = request.form

    message = data.get("message", "")

    if not isinstance(message, str):
        return json_error("Message must be text.", 400)

    message = message.strip()

    if not message:
        return json_error(
            "Please enter a message.",
            400
        )

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

            timeout=(15, 60)
        )

        if not response.ok:
            app.logger.error(
                "Groq error HTTP %s: %s",
                response.status_code,
                response.text[:1000]
            )

            return json_error(
                "Kintu AI could not answer. Please try again.",
                502
            )

        result = response.json()

        reply = result["choices"][0]["message"]["content"]

        if not isinstance(reply, str) or not reply.strip():
            return json_error(
                "Kintu AI returned an empty answer.",
                502
            )

        return Response(
            reply.strip(),
            status=200,
            mimetype="text/plain; charset=utf-8"
        )

    except requests.Timeout:
        app.logger.exception("Groq request timed out")

        return json_error(
            "Kintu AI took too long to respond.",
            504
        )

    except requests.RequestException:
        app.logger.exception("Groq connection failed")

        return json_error(
            "Connection problem. Please try again.",
            502
        )

    except (KeyError, IndexError, TypeError, ValueError):
        app.logger.exception("Unexpected Groq response")

        return json_error(
            "Unexpected AI response.",
            502
        )


# ==========================================
# IMAGE GENERATION - POLLINATIONS
# ==========================================

@app.route("/generate-image", methods=["POST"])
def generate_image():

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return json_error(
            'Send JSON containing a "prompt" field.',
            400
        )

    prompt = data.get("prompt", "")

    if not isinstance(prompt, str):
        return json_error("Prompt must be text.", 400)

    prompt = prompt.strip()

    if not prompt:
        return json_error(
            "Please describe the image you want.",
            400
        )

    if len(prompt) > 4000:
        return json_error(
            "Your image description is too long.",
            400
        )

    safe_prompt = urllib.parse.quote(prompt, safe="")

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
        f"?width={width}&height={height}"
        f"&seed={seed}&nologo=true&model=flux"
    )

    app.logger.info("Generated image URL: %s", image_url)

    return jsonify({
        "success": True,
        "image_url": image_url,
        "format": "png"
    }), 200


# ==========================================
# PHOTO SPY AND INTELLIGENCE - GEMINI
#
# Supports:
# 1. Kodular Web.PostFile raw image upload
# 2. Multipart image upload
# 3. JSON Base64 image
# 4. JSON image URL
# ==========================================

@app.route("/describe-image", methods=["POST"])
def describe_image():

    gemini_key = os.environ.get("GEMINI_API_KEY")

    if not gemini_key:
        app.logger.error("GEMINI_API_KEY is missing")

        return json_error(
            "Photo Spy is not configured. Add GEMINI_API_KEY.",
            500
        )

    # --------------------------------------
    # GET THE USER'S QUESTION
    # --------------------------------------

    prompt = (
        request.args.get("prompt", "")
        or request.form.get("prompt", "")
        or request.form.get("message", "")
    )

    data = request.get_json(silent=True)

    if not prompt and isinstance(data, dict):
        prompt = (
            data.get("prompt", "")
            or data.get("message", "")
        )

    if not isinstance(prompt, str):
        prompt = str(prompt)

    prompt = prompt.strip()

    if not prompt:
        prompt = "Describe this image clearly and accurately."

    # --------------------------------------
    # RECEIVE THE IMAGE
    # --------------------------------------

    image_bytes = None
    mime_type = ""

    # METHOD 1: MULTIPART FILE UPLOAD

    if request.files:

        file_obj = request.files.get("image")

        if file_obj is None:
            file_obj = next(
                iter(request.files.values()),
                None
            )

        if file_obj is not None:

            image_bytes = file_obj.read()

            mime_type = detect_image_mime(
                image_bytes,
                file_obj.mimetype or ""
            )

    # METHOD 2: BASE64 IMAGE

    if image_bytes is None and isinstance(data, dict):

        image_b64 = (
            data.get("image", "")
            or data.get("image_base64", "")
        )

        if isinstance(image_b64, str) and len(image_b64) > 100:

            if "," in image_b64 and "base64" in image_b64[:100]:

                header, image_b64 = image_b64.split(",", 1)

                if "png" in header.lower():
                    mime_type = "image/png"

                elif "webp" in header.lower():
                    mime_type = "image/webp"

                else:
                    mime_type = "image/jpeg"

            try:
                image_bytes = base64.b64decode(
                    image_b64,
                    validate=True
                )

            except (ValueError, TypeError):
                image_bytes = None

    # METHOD 3: IMAGE URL

    if image_bytes is None and isinstance(data, dict):

        image_url = (
            data.get("image_url", "")
            or request.args.get("image_url", "")
        )

        if (
            isinstance(image_url, str)
            and image_url.startswith("https://")
        ):

            try:
                img_resp = requests.get(
                    image_url,
                    timeout=(10, 25),
                    headers={
                        "User-Agent": "KintuAI/1.0"
                    }
                )

                if img_resp.ok:

                    image_bytes = img_resp.content

                    mime_type = img_resp.headers.get(
                        "Content-Type",
                        ""
                    )

            except requests.RequestException:
                app.logger.exception(
                    "Could not download image URL"
                )

    # METHOD 4: RAW FILE BODY FROM KODULAR POSTFILE
    #
    # Kodular Web.PostFile sends the selected file
    # as the request body, not necessarily as
    # multipart/form-data.
    #
    # Therefore request.files may be empty.
    # We read the raw image bytes here.

    if image_bytes is None and not request.files:

        raw_body = request.get_data(cache=True)

        if raw_body:

            # Do not mistake JSON for a raw image.
            if not raw_body.lstrip().startswith(b"{"):

                image_bytes = raw_body

                mime_type = request.headers.get(
                    "Content-Type",
                    ""
                )

    # --------------------------------------
    # CHECK THE RECEIVED PHOTO
    # --------------------------------------

    if not image_bytes:

        app.logger.error(
            "No image received. Content-Type: %s",
            request.content_type
        )

        return json_error(
            "No photo reached Kintu. Select the photo again and retry.",
            400
        )

    if len(image_bytes) > 10 * 1024 * 1024:

        return json_error(
            "Photo is too large. Please use an image under 10 MB.",
            413
        )

    # Detect the real image format.

    mime_type = detect_image_mime(
        image_bytes,
        mime_type
    )

    if not mime_type:

        return json_error(
            "Unsupported image. Please use JPG, PNG or WEBP.",
            415
        )

    app.logger.info(
        "Photo received successfully: %s bytes, %s",
        len(image_bytes),
        mime_type
    )

    # --------------------------------------
    # SEND IMAGE TO GEMINI
    # --------------------------------------

    image_bytes_b64 = base64.b64encode(
        image_bytes
    ).decode("ascii")

    model = os.environ.get(
        "GEMINI_MODEL",
        "gemini-3.8-flash"
    )

    try:

        gemini_url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/{model}:generateContent"
            f"?key={urllib.parse.quote(gemini_key, safe='')}"
        )

        vision_prompt = (
            "You are Kintu AI's Photo Spy and Intelligence assistant. "
            "Look carefully at the attached image and answer the user's "
            "question based on what is actually visible. "
            "If it is a document, read and explain the legible text. "
            "If some text is too small or unclear, say so instead of guessing. "
            "Be specific, accurate, friendly and useful. "
            "Follow Kintu AI's language policy.\n\n"
            "User's question: "
            + prompt
        )

        gemini_payload = {

            "system_instruction": {
                "parts": [
                    {
                        "text": SYSTEM_PROMPT
                    }
                ]
            },

            "contents": [
                {
                    "role": "user",

                    "parts": [

                        {
                            "text": vision_prompt
                        },

                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": image_bytes_b64
                            }
                        }

                    ]
                }
            ],

            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 1200
            }
        }

        g_resp = requests.post(
            gemini_url,
            json=gemini_payload,
            headers={
                "Content-Type": "application/json"
            },
            timeout=(20, 90)
        )

        # ----------------------------------
        # HANDLE GEMINI ERRORS
        # ----------------------------------

        if not g_resp.ok:

            app.logger.error(
                "Gemini error HTTP %s: %s",
                g_resp.status_code,
                g_resp.text[:1500]
            )

            if g_resp.status_code in (400, 404):

                return json_error(
                    "The Gemini model or request is unavailable. "
                    "Check your GEMINI_MODEL and API key.",
                    502
                )

            if g_resp.status_code == 429:

                return json_error(
                    "Photo Spy has reached its AI usage limit. Try again later.",
                    429
                )

            return json_error(
                "Kintu could not analyse the photo. Please try again.",
                502
            )

        # ----------------------------------
        # EXTRACT THE DESCRIPTION
        # ----------------------------------

        result = g_resp.json()

        candidates = result.get("candidates", [])

        description = ""

        if candidates:

            parts = (
                candidates[0]
                .get("content", {})
                .get("parts", [])
            )

            description = "".join(
                part.get("text", "")
                for part in parts
                if isinstance(part, dict)
            ).strip()

        if not description:

            app.logger.error(
                "Gemini returned no text: %s",
                str(result)[:1500]
            )

            return json_error(
                "Kintu received the photo but returned no description.",
                502
            )

        app.logger.info(
            "Photo description generated successfully."
        )

        return jsonify({

            "success": True,

            "description": description

        }), 200

    except requests.Timeout:

        app.logger.exception(
            "Gemini image request timed out"
        )

        return json_error(
            "Kintu took too long to analyse the photo. Please try again.",
            504
        )

    except requests.RequestException:

        app.logger.exception(
            "Gemini connection failed"
        )

        return json_error(
            "Could not connect to the image AI service.",
            502
        )

    except (ValueError, TypeError, KeyError):

        app.logger.exception(
            "Unexpected Gemini response"
        )

        return json_error(
            "Kintu received an unexpected AI response.",
            502
        )


# ==========================================
# DISPLAY OLD GENERATED IMAGES
# ==========================================

@app.route(
    "/generated-images/<image_id>.png",
    methods=["GET"]
)
def get_generated_image(image_id):

    with GENERATED_IMAGES_LOCK:

        image_bytes = GENERATED_IMAGES.get(image_id)

    if image_bytes is None:

        return Response(
            "Image not found. It may have expired.",
            status=404,
            mimetype="text/plain"
        )

    return Response(
        image_bytes,
        status=200,
        mimetype="image/png",
        headers={
            "Cache-Control": "public, max-age=3600",
            "X-Content-Type-Options": "nosniff"
        }
    )


# ==========================================
# FILE SIZE ERROR
# ==========================================

@app.errorhandler(413)
def request_too_large(error):

    return json_error(
        "Image is too large. Try a smaller photo.",
        413
    )


# ==========================================
# START SERVER
# ==========================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", "5000")
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
