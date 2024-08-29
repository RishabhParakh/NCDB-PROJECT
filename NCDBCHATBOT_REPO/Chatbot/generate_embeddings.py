import os
import json
import torch
from transformers import AutoTokenizer, AutoModel
from langchain.document_loaders import CSVLoader, PyMuPDFLoader, TextLoader
from langchain.docstore.document import Document
from typing import List

# Define the custom MiniLMSentenceEmbeddings class
class MiniLMSentenceEmbeddings:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return embeddings.tolist()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Define the function to load documents
def load_documents(file_path):
    ext = "." + file_path.rsplit(".", 1)[-1]
    print(f"Processing file: {file_path} with extension: {ext}")  # Debugging print statement
    if ext == ".csv":
        loader = CSVLoader(file_path)
    elif ext == ".pdf":
        loader = PyMuPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".json":
        return load_json_documents(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader.load()

# Custom function to load JSON documents
def load_json_documents(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    
    documents = []
    for idx, item in enumerate(json_data):
        content = f"question: {item['question']}; answer: {item['answer']}"
        metadata = {'source': os.path.basename(file_path), 'num': idx}
        documents.append(Document(page_content=content, metadata=metadata))
    
    return documents

# Directory containing your documents
source_directory = "source_documents"
embedding_model = MiniLMSentenceEmbeddings('sentence-transformers/all-MiniLM-L6-v2')

all_embeddings = []
all_metadata = []

# Process each file
for filename in os.listdir(source_directory):
    file_path = os.path.join(source_directory, filename)
    try:
        documents = load_documents(file_path)
        for doc in documents:
            embeddings = embedding_model.embed_documents([doc.page_content])
            all_embeddings.append(embeddings[0])
            all_metadata.append(doc.metadata)
            # Debugging output for each document processed
            print(f"Processed document: {doc.metadata['source']}, Content: {doc.page_content[:100]}...")  # Show first 100 characters for brevity
    except ValueError as e:
        print(e)  # Print the error for debugging

# Save the embeddings and metadata
with open('embeddings.json', 'w') as f:
    json.dump({'embeddings': all_embeddings, 'metadata': all_metadata}, f)

print("Embedding generation complete. Saved to embeddings.json")

