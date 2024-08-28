import warnings

warnings.simplefilter("ignore")

import json
import logging
import hashlib
import os
import pandas as pd
from typing import List
from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.contrib.admin.views.decorators import staff_member_required
from transformers import T5Tokenizer, T5ForConditionalGeneration
from EB.views import run_email_script
import socket
from pdfminer.high_level import extract_text
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer


logger = logging.getLogger(__name__)
dir(settings)

DEFAULT_RESPONSE = 'I am sorry, I do not understand. Please contact mrcadillac@newcadillacdatabase.org for further assistance.'

CHATBOT_QUESTIONS_URL_ROOT = settings.CHATBOT_QUESTIONS_URL_ROOT

questions_without_answer_path = os.path.join(CHATBOT_QUESTIONS_URL_ROOT, "questions_without_answer.json")
questions_with_answers_path = os.path.join(CHATBOT_QUESTIONS_URL_ROOT, "questions_with_answers.json")
source_documents_path = "/home/ncdbproj/NCDBContent/Chatbot/source/source_documents"
try:
# Load the fine-tuned T5 model and tokenizer
      model_path = "/home/ncdbproj/NCDBContent/Chatbot/finetuned_t5_model"
      tokenizer = T5Tokenizer.from_pretrained(model_path, use_fast=True)
      model = T5ForConditionalGeneration.from_pretrained(model_path)

except Exception:
                pass
# Load pre-defined questions and answers
with open(questions_with_answers_path, "r", encoding="utf-8") as saved_questions_file:
    shortcut_qa = json.load(saved_questions_file)

# Helper function to read PDF files
def read_pdf(file_path):
    text = extract_text(file_path)
    print(f"Extracted text from {file_path}: {text[:500]}")  # Print first 500 characters for verification
    return text

# Helper function to read PDF files and return pages
def read_pdf_pages(file_path):
    pages = []
    for page_layout in extract_pages(file_path):
        page_text = ''
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                page_text += element.get_text()
        pages.append(page_text)
    return pages

# Load source documents (JSON, PDF, CSV)
def load_source_documents(path):
    documents = {}
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        if file_name.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as json_file:
                documents[file_name] = json.load(json_file)
        elif file_name.endswith(".pdf"):
            documents[file_name] = read_pdf_pages(file_path)
        elif file_name.endswith(".csv"):
            documents[file_name] = pd.read_csv(file_path).to_dict(orient='records')
    return documents

source_documents = load_source_documents(source_documents_path)

# Debug: Print loaded documents to ensure the files are loaded correctly
print(f"Loaded documents: {source_documents.keys()}")

# Extract text from the PDF for querying
pdf_text_pages = source_documents.get('about-and-dream-cars-and-more.pdf', [])

def query_csv_data(model_name):
    if 'car_model.csv' not in source_documents:
        logger.error("car_model.csv not found in source documents")
        return None
    
    for record in source_documents['car_model.csv']:
        if record['cadillac_car_model'].lower() == model_name.lower():
            return record
    return None

def query_cadillac_category(category_name):
    if 'important-cadillac-categories.csv' not in source_documents:
        logger.error("important-cadillac-categories.csv not found in source documents")
        return None
    
    for record in source_documents['important-cadillac-categories.csv']:
        if record['Cadillac_related_category'].lower() == category_name.lower():
            return record
    return None

def query_pdf_text(query):
    print(f"Querying PDF text with: {query}")
    for page in pdf_text_pages:
        if query.lower() in page.lower():
            start_index = page.lower().index(query.lower())
            end_index = start_index + 200  # Return a snippet of 200 characters around the query
            answer = page[start_index:end_index]
            print(f"PDF contains relevant information: {answer}")
            return answer
    print("PDF does not contain relevant information.")
    return None

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def find_similar_question(query):
    best_score = float('inf')
    best_match = None
    for question in shortcut_qa:
        score = levenshtein_distance(query.lower(), question['question'].lower())
        if score < best_score:
            best_score = score
            best_match = question['question']
    return best_match

def chat_page(request):
    """ Chatbot demo page """
    return render(request, 'demo.html')

def send_alert_email(alert_subject, alert_body):
    try:
        logger.info("Email subject: " + alert_subject)
        logger.info("Email message: " + alert_body)

        script_args = []
        script_args.append(False)
        script_args.append(alert_subject)
        script_args.append(settings.EMAIL_ADMIN)
        script_args.append(alert_body)
        script_output = run_email_script(script_args)
        if script_output.stderr == '':
            print('The email was sent successfully.')
            logger.info('The email was sent successfully.')
        else:
            logger.info(f"The SendMail emailer script encountered an error: {script_output.stderr}")
    except Exception as e:
        logger.error(f'An error occurred running the SendMail emailer script: {e}')

def add_unanswered_question(question):
    """Add unanswered question to a file"""
    with open(questions_without_answer_path, 'r', encoding='utf-8') as json_file:
        questions_without_answer = json.load(json_file)
    question_dict = questions_without_answer["questions"]
    question_hash = hashlib.md5(question.encode("utf-8")).hexdigest()
    new_unaswered_question = {
        "content": question
    }
    question_dict[question_hash] = new_unaswered_question
    questions_without_answer["questions"] = question_dict
    print(f"len of question_dict is {len(question_dict)}")
    if len(question_dict) > 50:
        print("preparing to send email to admin")
        alert_subject = "Alert: There are more than 50 questions in NCDB waiting for you to answer"
        alert_body = "Message: Please answer new questions in https://www.newcadillacdatabase.org."
        send_alert_email(alert_subject, alert_body)
    with open(questions_without_answer_path, 'w', encoding='utf-8') as json_file:
        json.dump(questions_without_answer, json_file)

def generate_answer_from_llm(query):
    HOST = '0.0.0.0'
    PORT = 12345 

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(query.encode())
        response = s.recv(1024).decode()

    return response

def generate_response_from_model(query):
    inputs = tokenizer(query, return_tensors="pt",add_special_tokens=True)
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def combine_answers(answers: List[str]) -> str:
    prompt = "Combine the following information into a single clear answer, DO NOT CONCATENATE.\n"
    for answer in answers:
        prompt += f"{answer}\n"

    logger.info(f"Prompt sent to model: {prompt}")

    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    output = model.generate(inputs.input_ids, max_length=500, temperature=1, top_p=1)
    combined_response = tokenizer.decode(output[0], skip_special_tokens=True)

    logger.info(f"Combined response from model: {combined_response}")

    return combined_response.strip()

def chat(request):
    """ Answer the query from user """
    print("#################### I AM HERE (0) - chat CALLED!!!!!")
    query = request.GET.get("query")
    print(f"------------------------------------------ QUERY SUBMITTED: {query}")

    response_str = None
    answers = []

    logger.info(f"Received query: {query}")

    # Check if query matches a car model in the car_model CSV file
    car_model_data = query_csv_data(query)
    logger.info(f"Car model data found: {car_model_data}")
    if car_model_data:
        answers.append(f"{car_model_data['introduction']} More information can be found here: {car_model_data['image_and_description_link']}")

    # Check if query matches a category in the important-cadillac-categories CSV file
    category_data = query_cadillac_category(query)
    logger.info(f"Category data found: {category_data}")
    if category_data:
        answers.append(f"{category_data['Introduction/examples']} More information can be found here: {category_data['Link']}")

    # Check if query matches the PDF text
    pdf_response = query_pdf_text(query)
    logger.info(f"PDF response found: {pdf_response}")
    if pdf_response:
        answers.append(pdf_response)

    if not answers:
        # Find the most similar question
        similar_question = find_similar_question(query)
        matching_answers = [item['answer'] for item in shortcut_qa if item['question'].lower() == similar_question.lower()]

        logger.info(f"Matching answers found: {matching_answers}")

        if matching_answers:
            answers.extend(matching_answers)
        else:
            answer_generated_from_llm = generate_answer_from_llm(query)
            logger.info(f"Generated answer from LLM: {answer_generated_from_llm}")
            if answer_generated_from_llm == "I do not know.":
                response_str = DEFAULT_RESPONSE
                add_unanswered_question(query)
            else:
                answers.append(answer_generated_from_llm)

    if answers:
        response_str = combine_answers(answers)
        logger.info(f"Combined answers: {response_str}")

    resp_data = {
        "status": True,
        "data": response_str
    }
    return HttpResponse(json.dumps(resp_data))

@staff_member_required
def admin(request):
    """Chatbot admin page"""
    ctx = {}
    return render(request, 'Chatbot/admin.html', ctx)

@staff_member_required
def get_new_question(request):
    """handle get request for new questions"""
    if is_ajax(request) and request.method == "GET":
        empty_question = {"hash": "N/A", "content": "N/A"}
        with open(questions_without_answer_path, 'r', encoding='utf-8') as file:
            unaswered_questions = json.load(file)["questions"]
            if not bool(unaswered_questions):
                question_hash = "N/A"
            else:
                question_hash = next(iter(unaswered_questions))
        if question_hash == "N/A":
            question = empty_question
        else:
            question = {"hash": question_hash,
                        "content": unaswered_questions[question_hash]["content"]}
        logger.info("The new question sent has content: %s, and hash: %s",
                    question["content"], question["hash"])
        return JsonResponse({"question": question}, status=200)
    else:
        return JsonResponse({"error": "Wrong request"}, status=400)

def is_ajax(request):
    """helper function to check whether a request is a ajax request"""
    return request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest'

@csrf_exempt
@staff_member_required
def answer(request):
    """handle request to post an answer to a question"""
    if is_ajax(request) and request.method == "POST":
        question_with_answer = json.loads(request.body).get("question_with_answer")
        logger.info(question_with_answer)
        question_answer = question_with_answer.get("answer")
        question_hash = question_with_answer.get("hash")
        question_content = question_with_answer.get("question")
        msg = "We get the answer: " + question_answer + " for question: " + question_content + " with hash: " + question_hash
        logger.info(msg)

        # delete this question in questions_without_answer_path
        with open(questions_without_answer_path, "r", encoding="utf-8") as file:
            questions_without_answer = json.load(file)
            del questions_without_answer["questions"][question_hash]

        # Load existing data from the JSON file
        with open(questions_with_answers_path, 'r') as json_file:
            existing_data = json.load(json_file)

        # New question and answer to add
        new_entry = {
            "question": question_content,
            "answer": question_answer
        }

        # Add the new entry at the beginning of the existing data
        existing_data.insert(0, new_entry)

        # Write the updated data back to the JSON file
        with open(questions_with_answers_path, 'w') as json_file:
            json.dump(existing_data, json_file, indent=4)

        print("New entry added to the beginning of the JSON file.")
        return JsonResponse({"msg": msg}, status=200)
    else:
        return JsonResponse({"error": "Wrong request"}, status=400)

