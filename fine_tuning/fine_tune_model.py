from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Load SAMSum dataset from Hugging Face's datasets library with trust_remote_code=True
dataset = load_dataset("samsum", trust_remote_code=True)

# Load the pre-trained model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Preprocess the data for summarization
def preprocess_data(examples):
    inputs = [f"summarize: {dialogue}" for dialogue in examples["dialogue"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, return_tensors="pt", padding="max_length")

    # Prepare target labels
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=150, truncation=True, return_tensors="pt", padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize the dataset
tokenized_datasets = dataset.map(preprocess_data, batched=True)

# Define a custom collate function for handling padding and batching
def collate_fn(batch):
    input_ids = torch.stack([torch.tensor(item['input_ids']) if isinstance(item['input_ids'], list) else item['input_ids'] for item in batch])
    attention_mask = torch.stack([torch.tensor(item['attention_mask']) if isinstance(item['attention_mask'], list) else item['attention_mask'] for item in batch])
    labels = torch.stack([torch.tensor(item['labels']) if isinstance(item['labels'], list) else item['labels'] for item in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

# Create a DataLoader for training with the custom collate_fn
train_loader = DataLoader(tokenized_datasets["train"], batch_size=8, shuffle=True, collate_fn=collate_fn)

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=3e-5)

# Fine-tune the model
epochs = 3  # You can adjust the number of epochs based on your needs
model.train()

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        # Move input data to the device (GPU/CPU)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Display loss
        loop.set_postfix(loss=loss.item())

# Save the fine-tuned model
model.save_pretrained("./app/model/fine_tuned_t5_samsum")
tokenizer.save_pretrained("./app/model/fine_tuned_t5_samsum")

print("Model fine-tuning complete and saved.")
