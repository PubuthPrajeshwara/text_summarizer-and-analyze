from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertForTokenClassification, DistilBertTokenizer, DistilBertForSequenceClassification, BartForConditionalGeneration, BartTokenizer
import torch
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

# Load the T5 summarization model and tokenizer
t5_model_name = './app/model/fine_tuned_t5_samsum'
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

# Load the BERT keyword extraction model and tokenizer
bert_model_name = './app/model/fine_tuned_bert_keyword_extraction'
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertForTokenClassification.from_pretrained(bert_model_name)

# Load the fine-tuned sentiment analysis model and tokenizer for DistilBERT
sentiment_model_name = './app/model/fine_tuned_sentiment_model'
sentiment_tokenizer = DistilBertTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = DistilBertForSequenceClassification.from_pretrained(sentiment_model_name)

# Load the pre-trained BART model and tokenizer for topic generation
bart_model_name = "facebook/bart-large-cnn"
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name)

# Set up the device for both models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t5_model = t5_model.to(device)
bert_model = bert_model.to(device)
sentiment_model = sentiment_model.to(device)
bart_model = bart_model.to(device)

# Function to summarize text using T5
def summarize_text(text):
    inputs = t5_tokenizer.encode(f"summarize: {text}", return_tensors="pt", max_length=512, truncation=True)
    inputs = inputs.to(device)
    summary_ids = t5_model.generate(inputs, max_length=150, min_length=40, num_beams=4, length_penalty=2.0, early_stopping=True)
    return t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Load label list
label_list = ["O"] + [
    "B-MISC", "I-MISC",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC"
]

# Function to extract keywords using BERT
def extract_keywords(text):
    with torch.no_grad():
        inputs = bert_tokenizer.encode_plus(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            is_split_into_words=False
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()

        tokens = bert_tokenizer.convert_ids_to_tokens(input_ids[0])
        keywords = []
        current_keyword = []

        for token, pred in zip(tokens, predictions):
            label = label_list[pred]
            # Debugging line for token prediction
            logger.debug(f"Token: {token}, Prediction: {label}")

            if label.startswith("B-"):
                if current_keyword:
                    keywords.append(" ".join(current_keyword))
                    current_keyword = []
                # Handle special tokens without '##'
                if token in bert_tokenizer.all_special_tokens:
                    current_keyword.append(token)
                else:
                    current_keyword.append(token.replace("##", ""))
            elif label.startswith("I-") and current_keyword:
                current_keyword.append(token.replace("##", ""))
            else:
                if current_keyword:
                    keywords.append(" ".join(current_keyword))
                    current_keyword = []

        if current_keyword:
            keywords.append(" ".join(current_keyword))

        # Clean and deduplicate keywords
        keywords = [kw.strip() for kw in keywords]
        keywords = list(set(keywords))

        if not keywords:
            logger.debug("No keywords extracted.")

        return keywords

# Function to perform sentiment analysis
def analyze_sentiment(text):
    # Tokenize and encode the input text
    inputs = sentiment_tokenizer.encode_plus(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Get model predictions
    with torch.no_grad():
        outputs = sentiment_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

    # Map predictions to sentiment labels (assuming binary classification: 0=Negative, 1=Positive)
    sentiment_label = "Positive" if predictions[0] == 1 else "Negative"

    return sentiment_label


# Function to generate topics using BART
def generate_topic(input_text):
    # Tokenize and encode the input text
    inputs = bart_tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = inputs.to(device)

    # Generate a summary (topic)
    summary_ids = bart_model.generate(
        inputs["input_ids"],
        max_length=10,  # Limit the output length for topic generation
        min_length=2,  # Set a min_length to avoid warning (adjust as needed)
        num_beams=4,  # Beam search to improve the quality of generation
        early_stopping=True
    )

    # Decode the output
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    logger.debug(f"Generated summary: {summary}")

    return summary