import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from groq import Groq

load_dotenv()

app = Flask(__name__)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def summarize_text(text):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You summarize uploaded text clearly and concisely."
            },
            {
                "role": "user",
                "content": f"Summarize this text in 1-2 paragraphs: \n\n{text}"
            }
        ],
        temperature=0.3,
        max_tokens=500
    )
    return response.choices[0].message.content


@app.route("/", methods=["GET", "POST"])
def index():
    original_text = ""
    summary = ""
    error = ""

    if request.method == "GET":
        return render_template(
            "index.html", original_text=original_text, summary=summary, error=error
        )

    uploaded_file = request.files.get("file")
    if not uploaded_file:
        error = "Please upload a TXT file."
        return render_template(
            "index.html", original_text=original_text, summary=summary, error=error
        ), 400

    filename = uploaded_file.filename or ""
    if not filename.lower().endswith(".txt"):
        error = "Only TXT files are allowed."
        return render_template(
            "index.html", original_text=original_text, summary=summary, error=error
        ), 400

    try:
        original_text = uploaded_file.read().decode("utf-8").strip()
    except UnicodeDecodeError:
        error = "File must be UTF-8 encoded text."
        return render_template(
            "index.html", original_text=original_text, summary=summary, error=error
        ), 400

    if not original_text:
        error = "The uploaded file is empty."
        return render_template(
            "index.html", original_text=original_text, summary=summary, error=error
        ), 400

    try:
        summary = summarize_text(original_text)
    except Exception as e:
        error = f"Failed to summarize text: {e}"
        return render_template(
            "index.html", original_text=original_text, summary=summary, error=error
        ), 500

    return render_template(
        "index.html", original_text=original_text, summary=summary, error=error
    )


@app.route("/health")
def health():
    has_key = bool(os.getenv("GROQ_API_KEY"))
    return jsonify({"status": "ok", "groq_key_loaded": has_key})


if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY missing in .env")
    app.run(debug=True)
