from transformers import T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration, LEDTokenizer, LEDForConditionalGeneration
import torch

# Function to detect the type of text: research paper, news article, or general paragraph
def detect_text_type(text):
    # Convert to lowercase for easier comparison
    text_lower = text.lower()

    # Heuristics for research paper
    if any(word in text_lower for word in ['introduction', 'abstract', 'methodology', 'results', 'discussion', 'references', '[', 'et al']):
        return 'research_paper'
    
    # Heuristics for news article
    elif any(word in text_lower for word in ['news', 'reporter', 'journalist', 'breaking', 'headline', 'cnn', 'bbc']):
        return 'news_article'
    
    # Default to general text
    return 'general'

# Define the model loading logic based on the task type (research, news, or general)
def load_model_for_task(task_type):
    if task_type == 'research_paper':
        model_name = 'allenai/led-large-16384-arxiv'  # Model for research paper summarization
        tokenizer = LEDTokenizer.from_pretrained(model_name)
        model = LEDForConditionalGeneration.from_pretrained(model_name)
    elif task_type == 'news_article':
        model_name = 'facebook/bart-large-cnn'  # Model for news articles
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)
    else:
        model_name = 't5-large'  # Default model for general summarization
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Move model to the appropriate device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return tokenizer, model, device

# Function to summarize text based on the detected text type
def summarize_text(text):
    # Detect the type of text (research paper, news article, or general)
    task_type = detect_text_type(text)
    
    # Load the appropriate model for the task
    tokenizer, model, device = load_model_for_task(task_type)
    
    # Tokenize the input and move to device
    inputs = tokenizer.encode(f"summarize: {text}", return_tensors="pt", max_length=1024, truncation=True)
    inputs = inputs.to(device)
    
    # Generate the summary using the model
    summary_ids = model.generate(inputs, max_length=400, min_length=40, num_beams=4, length_penalty=2.0, early_stopping=True)
    
    # Decode and return the summary
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)



# Load the model and tokenizer
#model_name = './app/model/fine_tuned_t5_samsum'  # Path where model is saved
#tokenizer = T5Tokenizer.from_pretrained(model_name)
#model = T5ForConditionalGeneration.from_pretrained(model_name)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = model.to(device)

#def summarize_text(text):
#    inputs = tokenizer.encode(f"summarize: {text}", return_tensors="pt", max_length=512, truncation=True)
#    inputs = inputs.to(device)
#    summary_ids = model.generate(inputs, max_length=150, min_length=40, num_beams=4, length_penalty=2.0, early_stopping=True)
#    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
