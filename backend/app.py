import os
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import PyPDF2
import io

# Serve frontend from ../frontend relative to this file
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

# ── IBM watsonx configuration ──────────────────────────────────────────────
WATSONX_URL = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
MODEL_ID    = "ibm/granite-4-h-small"
PROJECT_ID  = "44894380-bf77-417d-a80b-c20cd6983a5e"
API_KEY     = os.environ.get("WATSONX_API_KEY", "1cc63859-e327-4503-a906-f0c90ce56225")

IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"


def get_iam_token(api_key: str) -> str:
    """Exchange IBM Cloud API key for a Bearer token."""
    response = requests.post(
        IAM_TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": api_key,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract all text from a PDF file bytes object."""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text.strip())
    return "\n\n".join(text_parts)


def chunk_text(text: str, max_chars: int = 6000) -> list:
    """Split large text into chunks so each fits within the model context."""
    paragraphs = text.split("\n\n")
    chunks, current = [], []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > max_chars and current:
            chunks.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(para)
        current_len += len(para)

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def summarise_chunk(token: str, chunk: str) -> str:
    """Send one text chunk to watsonx and return the summary."""
    prompt = (
        "You are an expert summariser. Read the following document excerpt and produce "
        "a concise bullet-point summary of the most important points. "
        "Use clear, plain English. Output only the bullet points, no preamble.\n\n"
        f"DOCUMENT:\n{chunk}\n\nSUMMARY:"
    )

    payload = {
        "model_id": MODEL_ID,
        "project_id": PROJECT_ID,
        "input": prompt,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 512,
            "min_new_tokens": 50,
            "repetition_penalty": 1.1,
        },
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    response = requests.post(WATSONX_URL, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    results = response.json().get("results", [])
    if results:
        return results[0].get("generated_text", "").strip()
    return ""


def merge_summaries(token: str, summaries: list) -> str:
    """If multiple chunk summaries exist, merge them into one final summary."""
    if len(summaries) == 1:
        return summaries[0]

    combined = "\n\n".join(summaries)
    prompt = (
        "You are an expert summariser. Below are partial summaries of sections of a large document. "
        "Combine them into a single cohesive bullet-point summary of the most important points. "
        "Remove duplicates. Output only the bullet points.\n\n"
        f"PARTIAL SUMMARIES:\n{combined}\n\nFINAL SUMMARY:"
    )

    payload = {
        "model_id": MODEL_ID,
        "project_id": PROJECT_ID,
        "input": prompt,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 700,
            "min_new_tokens": 50,
            "repetition_penalty": 1.1,
        },
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    response = requests.post(WATSONX_URL, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    results = response.json().get("results", [])
    if results:
        return results[0].get("generated_text", "").strip()
    return combined


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/summarise", methods=["POST"])
def summarise():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send a PDF as 'file'."}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported."}), 400

    try:
        pdf_bytes = file.read()
        text = extract_text_from_pdf(pdf_bytes)
        if not text.strip():
            return jsonify({"error": "Could not extract text from this PDF."}), 422

        token = get_iam_token(API_KEY)
        chunks = chunk_text(text)
        chunk_summaries = [summarise_chunk(token, c) for c in chunks]
        final_summary = merge_summaries(token, chunk_summaries)

        return jsonify(
            {
                "summary": final_summary,
                "pages": len(PyPDF2.PdfReader(io.BytesIO(pdf_bytes)).pages),
                "chunks_processed": len(chunks),
            }
        )

    except requests.HTTPError as e:
        return jsonify({"error": f"watsonx API error: {e.response.text}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
