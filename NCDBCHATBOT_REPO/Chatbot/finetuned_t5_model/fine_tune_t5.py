import os
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW

def fine_tune_t5():
    data = [
        {"input": "Answer 1: I am Rishabh. Answer 2: I am Sagar.", "output": "I am Rishabh Sagar."},
        {"input": "Answer 1: I like apples. Answer 2: I like oranges.", "output": "I like apples and oranges."},
        {"input": "Answer 1: My name is John. Answer 2: I work as a developer. Answer 3: I live in New York.", "output": "My name is John. I work as a developer and live in New York."},
        {"input": "Answer 1: I enjoy reading. Answer 2: I love hiking.", "output": "I enjoy reading and love hiking."},
        {"input": "Answer 1: My favorite color is blue. Answer 2: I love the ocean.", "output": "My favorite color is blue, and I love the ocean."},
        {"input": "Answer 1: I have a dog. Answer 2: His name is Max.", "output": "I have a dog named Max."},
        {"input": "Answer 1: I am a teacher. Answer 2: I teach math.", "output": "I am a math teacher."},
        {"input": "Answer 1: I like pizza. Answer 2: I like pasta.", "output": "I like pizza and pasta."},
        {"input": "Answer 1: I enjoy swimming. Answer 2: I swim every weekend.", "output": "I enjoy swimming and swim every weekend."},
        {"input": "Answer 1: I am learning Python. Answer 2: I want to build AI models.", "output": "I am learning Python to build AI models."},
        {"input": "Answer 1: I traveled to Paris. Answer 2: I visited the Eiffel Tower.", "output": "I traveled to Paris and visited the Eiffel Tower."},
        {"input": "Answer 1: I have a sister. Answer 2: She is younger than me.", "output": "I have a younger sister."},
        {"input": "Answer 1: I work at a tech company. Answer 2: I am a software engineer.", "output": "I work as a software engineer at a tech company."},
        {"input": "Answer 1: I love coffee. Answer 2: I drink it every morning.", "output": "I love coffee and drink it every morning."},
        {"input": "Answer 1: I play guitar. Answer 2: I have been playing for 5 years.", "output": "I have been playing guitar for 5 years."},
    ]

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    inputs = [item['input'] for item in data]
    targets = [item['output'] for item in data]

    max_length = 256  # Reduce sequence length to save memory
    tokenized_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
    tokenized_targets = tokenizer(targets, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['labels'] = self.labels['input_ids'][idx]
            return item

        def __len__(self):
            return len(self.labels['input_ids'])

    dataset = CustomDataset(tokenized_inputs, tokenized_targets)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Minimize batch size

    optimizer = AdamW(model.parameters(), lr=2e-5)

    model.train()
    accumulation_steps = 8  # Accumulate gradients over more steps

    num_epochs = 20  # Increase the number of epochs
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        for i, batch in enumerate(dataloader):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss = loss / accumulation_steps
            loss.backward()
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        print(f"Epoch {epoch + 1} completed with loss: {loss.item()}")

    # Save the fine-tuned model and tokenizer to the specified directory path
    save_directory = "/home/ncdbproj/CadillacDBProj/Chatbot/finetuned_t5_model"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Save the model
    model.save_pretrained(save_directory)

    # Save the tokenizer
    tokenizer.save_pretrained(save_directory)

    print("Model and tokenizer saved successfully.")

# Run the fine-tuning process
fine_tune_t5()

