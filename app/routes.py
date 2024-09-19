from flask import request, jsonify, render_template
from app import app
from app.model.model import summarize_text

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
