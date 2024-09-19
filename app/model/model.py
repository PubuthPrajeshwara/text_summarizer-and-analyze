from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load the model and tokenizer
model_name = './app/model/fine_tuned_t5_samsum'  # Path where model is saved
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def summarize_text(text):
    inputs = tokenizer.encode(f"summarize: {text}", return_tensors="pt", max_length=512, truncation=True)
    inputs = inputs.to(device)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, num_beams=4, length_penalty=2.0, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
