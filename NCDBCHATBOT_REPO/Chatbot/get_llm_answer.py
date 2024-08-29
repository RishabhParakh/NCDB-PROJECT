import time
import os
import logging
import numpy as np
from ingest import MiniLMSentenceEmbeddings
from langchain.vectorstores import Chroma
import os
import time
from chromadb.config import Settings
import signal
import subprocess

from dotenv import load_dotenv
import transformers

import socket

HOST = 'localhost'
PORT = 12345

load_dotenv()

# Load environment variables
logger = logging.getLogger(__name__)
time.clock = time.time

def process_query(query):

    response = 'I do not know.'

    # get the best result from similarity search from db
    search_results = db.similarity_search(query, k=5)  # Retrieve top 5 similar documents
    best_match = None
    highest_score = -1

    for result in search_results:
        # Calculate score or confidence (assuming higher score is better)
        score = calculate_score(result)
        if score > highest_score:
            highest_score = score
            best_match = result

    if best_match:
        context = best_match.page_content
        full_prompt = f"Context: {context}\n Only use the Context to construct one or more sentences for answer. You may or may not need to include 'yes' or 'no' and answer step by step. If the question cannot be answered, say 'I do not know.'. Question: What are the sentence or sentences that respond '{query}'? "
        
        input_ids = tokenizer.encode(full_prompt, return_tensors="pt")
        output = model.generate(input_ids, max_new_tokens=5000, return_dict_in_generate=True, output_scores=True)
        
        transition_scores = model.compute_transition_scores(
            output.sequences, output.scores, normalize_logits=True
        )
        
        num_scores = len(transition_scores[0])
        middle_scores = transition_scores[0][1:num_scores - 1] if num_scores > 2 else transition_scores[0][:1]
        
        if len(middle_scores) > 0:
            average_score = sum(middle_scores) / len(middle_scores)
            average_exp_score = np.exp(average_score)
            confident = average_exp_score > 0.5
            if confident:
                response = tokenizer.decode(output[0][0], skip_special_tokens=True)
    
    return response

def calculate_score(result):
    # Implement scoring logic based on your criteria
    # This could be based on embedding similarity, context length, or any other metric
    return result.metadata.get('score', 1)  # Example: use 'score' from metadata if available

def delRunningProcess():
    # Name of the process to search for
    process_name = "get_llm_answer.py"

    # Check if another instance is already running
    grep_cmd = f"ps aux | grep {process_name}"
    grep_process = subprocess.run(grep_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if grep_process.returncode == 0:
        output_lines = grep_process.stdout.strip().split('\n')
        for line in output_lines:
            columns = line.split()
            if process_name in columns[-1] and "grep" not in columns:
                old_pid = int(columns[1])
                if old_pid != os.getpid(): # don't kill current
                    try:
                        os.kill(old_pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass

delRunningProcess()

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=os.environ.get('PERSIST_DIRECTORY'),
        anonymized_telemetry=False
)
# Create the custom embeddings instance
embeddings_model = MiniLMSentenceEmbeddings(os.environ.get('EMBEDDING_MODEL_PATH'))
db = Chroma(persist_directory=os.environ.get('PERSIST_DIRECTORY'), embedding_function=embeddings_model, client_settings=CHROMA_SETTINGS)

questions_without_answer_path = os.path.join(os.environ.get('QUESTIONS_WITHOUT_ANSWER_PATH'))
questions_with_answers_path = os.path.join(os.environ.get('QUESTIONS_WITH_ANSWERS_PATH'))

model_name = "Google/flan-t5-base"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()

    while True:
        conn, addr = s.accept()
        with conn:
            query = conn.recv(1024).decode()
            if not query:
                break
            response = process_query(query)
            conn.sendall(response.encode())

