import os
from flask import Flask, request, Response, jsonify
from openai import OpenAI

app = Flask(__name__)

@app.get('/')
def health():
    return jsonify(status='ok', service='Kintu AI backend', endpoint='/chat')

@app.post('/chat')
def chat():
    # Supports Kodular form data and JSON requests.
    data = request.get_json(silent=True) or request.form or request.values
    message = (data.get('message') or '').strip()
    if not message:
        return Response('Please enter a message.', status=400, mimetype='text/plain')

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return Response('Backend setup error: OPENAI_API_KEY is not configured.', status=500, mimetype='text/plain')

    try:
        client = OpenAI(api_key=api_key)
        result = client.responses.create(
            model=os.getenv('OPENAI_MODEL', 'gpt-4.1-mini'),
            instructions=(
                'You are Kintu AI, a friendly assistant for users in Uganda. '
                'Reply mainly in natural Luganda, using a little British English when helpful. '
                'Answer in the language the user uses, be respectful and clear, and do not claim to be human.'
            ),
            input=message,
        )
        return Response(result.output_text or 'Nsonyiwa, ddamu ogezeeko.', mimetype='text/plain; charset=utf-8')
    except Exception:
        app.logger.exception('Kintu AI request failed')
        return Response('Waliwo ekizibu. Gezaako nate.', status=502, mimetype='text/plain; charset=utf-8')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', '5000')))
