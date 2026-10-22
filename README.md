# Kintu AI backend for Render

This is a small Flask backend for the Kintu AI Kodular app. It accepts a message at `POST /chat` as either form data (`message=...`) or JSON (`{"message":"..."}`) and returns the assistant reply as plain text, suitable for Kodular `Web1.GotText` using `responseContent`.

## Deploy on Render

1. Upload these three files (`app.py`, `requirements.txt`, `README.md`) to a GitHub repository.
2. In Render, choose **New + → Web Service** and connect that repository.
3. Use these settings:
   - Runtime: Python
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app`
4. In the Render service's **Environment** settings, add `OPENAI_API_KEY` with your own secret API key. Optionally add `OPENAI_MODEL` (default: `gpt-4.1-mini`). Never put the API key in Kodular blocks or the app.
5. Deploy. Render will assign a real service URL, for example `https://your-service-name.onrender.com`.
6. In Kodular, set `Web1.Url` to the real URL plus `/chat`, e.g. `https://your-service-name.onrender.com/chat`.

## Kodular request

For a form-encoded request, use `Web1.BuildRequestData` with a list containing the pair `message` → `TxtMessage.Text`, then pass the result to `Web1.PostText`. This backend accepts that format. In `Web1.GotText`, if `responseCode = 200`, set your reply label's Text to `responseContent`.

## Check deployment

Open the service root URL in a browser. It should return JSON with `status: ok`. This only checks that the service is running; chat also requires a valid API key and available model access.
