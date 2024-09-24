from flask import request, jsonify, render_template
from app import app
from app.model.model import summarize_text
import PyPDF2
import pdfplumber
import os

@app.route('/')
def index():
    # Render the index.html from the templates directory
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    text = data.get('text')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    summary = summarize_text(text)
    return jsonify({'summary': summary})

# Route for PDF-based summarization
@app.route('/summarize_pdf', methods=['POST'])
def summarize_pdf():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No PDF file provided'}), 400

    pdf_file = request.files['pdf']

    # Save the uploaded file to a temporary location
    pdf_path = os.path.join("temp_files", pdf_file.filename)
    pdf_file.save(pdf_path)

    # Extract text from the PDF
    extracted_text = extract_text_from_pdf(pdf_path)

    # Generate summary for the extracted text
    summary = summarize_text(extracted_text)

    # Optionally, remove the temporary file after processing
    os.remove(pdf_path)

    return jsonify({'summary': summary})

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF using pdfplumber or PyPDF2.
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()

    # Alternatively, you could use PyPDF2:
    # with open(pdf_path, 'rb') as f:
    #     reader = PyPDF2.PdfReader(f)
    #     for page in reader.pages:
    #         text += page.extract_text()

    return text
