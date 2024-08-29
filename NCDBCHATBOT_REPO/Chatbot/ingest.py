#!/usr/bin/env python3
import os
import subprocess
import glob
from typing import List
from multiprocessing import Pool
from tqdm import tqdm
from chromadb.config import Settings
from pathlib import Path
from dotenv import load_dotenv
import signal
import time

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

import csv
import json
from langchain.document_loaders import CSVLoader, PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

persist_directory = "db"
source_directory = "source_documents"

class MiniLMSentenceEmbeddings():
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

chunk_size = 600
chunk_overlap = 200

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=persist_directory,
        anonymized_telemetry=False
)

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".json": () # use customized method to load json
}

def clear_old_embeddings(directory_name):
    try:
        # Get the current script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct the full path to the directory
        dir_to_delete = os.path.join(script_dir, directory_name)
        
        # Check if the directory exists
        if os.path.exists(dir_to_delete):
            # Remove the directory and its contents
            os.system(f"rm -r {dir_to_delete}")
            print(f"Directory '{dir_to_delete}' and its contents deleted successfully.")
        else:
            print(f"Directory '{dir_to_delete}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext == ".json":
        with open(file_path, 'r') as file:
            json_data = json.load(file)

        # Extract questions and answers from the JSON
        documents = []
        for idx, question_data in enumerate(json_data):
            content = f"question: {question_data['question']}; answer: {question_data['answer']}"
            metadata = {'source': os.path.basename(file_path), 'num': idx}
            document = Document(page_content=content, metadata=metadata)
            documents.append(document)
        return documents

    if os.path.basename(file_path) == "car_model.csv" or os.path.basename(file_path) == "important-cadillac-categories.csv":
        documents = []
        with open(file_path, 'r') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row_number, row in enumerate(csvreader, start=1):
                if os.path.basename(file_path) == "car_model.csv":
                    content = f"Answer that respond {row['cadillac_car_model']}: {row['introduction']}, please see link: {row['image_and_description_link']})"
                elif os.path.basename(file_path) == "important-cadillac-categories.csv":
                    content = f"{row['Introduction/examples']} and more {row['Cadillac_related_category'].lower()} at link: {row['Link']}"
                metadata = {'source': os.path.basename(file_path), 'num': row_number}
                document = Document(page_content=content, metadata=metadata)
                documents.append(document)
            return documents
   
    elif ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()
    
    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
        
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.extend(docs)
                pbar.update()
  
    return results

def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts

def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False

def restart_llm_process():
    try:
        venvpath = os.path.join(os.environ.get('VENV_PATH', ''), 'activate')
        scriptDirectory=os.environ.get('SCRIPT_DIRECTORY')
        scriptName = 'get_llm_answer.py'
        cmd = f"source {venvpath}; cd {scriptDirectory}; python {scriptName}"
        
        # Set timeout and start the process
        process = subprocess.Popen(cmd, shell=True, executable='/bin/bash')
        start_time = time.time()
        
        while process.poll() is None:
            elapsed_time = time.time() - start_time
            if elapsed_time > 60:  # If the process runs for more than 1 minute, terminate it
                process.terminate()
                raise TimeoutError("The subprocess took too long and was terminated.")
            time.sleep(1)
        
        completed_process = process.communicate()
        print(f"completed_process.returncode is {process.returncode}")
        return process.returncode
    except Exception as e:
        print(f"Error restarting process: {e}")
        return -1

def main():
    # Clear old embeddings, so later db directory can include latest embeddings for all documents
    clear_old_embeddings("db")
    
    # Create the custom embeddings instance
    embeddings = MiniLMSentenceEmbeddings('embedding_model')

    print(f"embeddings is {embeddings}")
    if does_vectorstore_exist(persist_directory):
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        collection = db.get()
        texts = process_documents([metadata['source'] for metadata in collection['metadatas']])
        print(f"Creating embeddings. May take some minutes...")
        db.add_documents(texts)
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        texts = process_documents()
        print(f"Creating embeddings. May take some minutes...")
        db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None
    print(f"Prepared latest documents.")

    # Restart get_answer_from_llm.py process
    restart_llm_process()

if __name__ == "__main__":
    main()

