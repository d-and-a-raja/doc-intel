import os
import json
from flask import Flask, request, render_template, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv import load_dotenv
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential

load_dotenv()

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB limit

ALLOWED_EXTENSIONS = {"pdf", "jpg", "jpeg", "png", "tiff", "bmp"}

client = DocumentIntelligenceClient(
    endpoint=os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"),
    credential=AzureKeyCredential(os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")),
)

CUSTOM_MODEL_ID = os.getenv("CUSTOM_MODEL_ID", "purchase-order-model-v2").strip().strip('"')


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def analyze_bytes(file_bytes, content_type):
    poller = client.begin_analyze_document(
        CUSTOM_MODEL_ID,
        body=file_bytes,
        content_type=content_type,
    )
    result = poller.result()

    all_docs = []
    for doc in result.documents:
        doc_data = {
            "doc_type": doc.doc_type,
            "confidence": round(doc.confidence * 100, 1),
            "fields": {},
        }
        for field_name, field_value in doc.fields.items():
            if field_value is None:
                doc_data["fields"][field_name] = {"value": "—", "confidence": None}
            else:
                conf = round(field_value.confidence * 100, 1) if field_value.confidence is not None else None
                doc_data["fields"][field_name] = {
                    "value": field_value.content or "—",
                    "confidence": conf,
                }
        all_docs.append(doc_data)

    return all_docs


MIME_MAP = {
    "pdf": "application/pdf",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "tiff": "image/tiff",
    "bmp": "image/bmp",
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    ext = file.filename.rsplit(".", 1)[1].lower()
    content_type = MIME_MAP.get(ext, "application/octet-stream")

    try:
        file_bytes = file.read()
        results = analyze_bytes(file_bytes, content_type)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
