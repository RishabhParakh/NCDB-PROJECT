from django.conf import settings
import json
import os
from chatterbot import ChatBot, comparisons, response_selection
from chatterbot.trainers import ListTrainer
import datetime

DEFAULT_RESPONSE = """I am sorry, I do not understand. 
        Please contact mrcadillac@newcadillacdatabase.org 
        for further assistance."""
MAXIMUM_SIMILARITY_THRESHOLD = 0.2


def train_bot():
    "Train bot using new questions with answers"
    unsaved_questions_with_answers_path = os.path.join(
        settings.BASE_DIR, 'static/', "chatbot_questions/unsaved_questions_with_answers.json")
    with open(unsaved_questions_with_answers_path, "r", encoding="utf-8") as unsaved_questions_file:
        unsaved_questions_with_answers = json.load(unsaved_questions_file)
        questions_answers = list(
            unsaved_questions_with_answers["questions"].values())
        training_list = [[q["content"], q["answer"]]
                         for q in questions_answers]
        chatbot = ChatBot('NCDB Chatbot',
                          storage_adapter='chatterbot.storage.SQLStorageAdapter',
                          logic_adapters=[
                              {
                                  'import_path': 'chatterbot.logic.BestMatch',
                                  "statement_comparison_function": comparisons.LevenshteinDistance,
                                  "response_selection_method": response_selection.get_first_response,
                                  'default_response': DEFAULT_RESPONSE,
                                  'threshold': 0.5,
                                  'maximum_similarity_threshold': MAXIMUM_SIMILARITY_THRESHOLD
                              }
                          ],
                          # preprocessors=['chatterbot.preprocessors.clean_whitespace'],
                          read_only=True
                          )
        trainer = ListTrainer(chatbot)
        for question_and_ans in training_list:
            trainer.train(question_and_ans)


def add_new_questions_with_answers():
    """Add new questions with answers to question corpus"""
    train_bot()
    unsaved_questions_with_answers_path = os.path.join(
        settings.BASE_DIR, 'static/', "chatbot_questions/unsaved_questions_with_answers.json")
    questions_with_answers_path = os.path.join(
        settings.BASE_DIR, 'static/', "chatbot_questions/questions_with_answers.json")
    with open(unsaved_questions_with_answers_path, "r", encoding="utf-8") as unsaved_questions_file:
        unsaved_questions_with_answers = json.load(unsaved_questions_file)
    with open(questions_with_answers_path, "r", encoding="utf-8") as questions_file:
        questions_with_answers = json.load(questions_file)
    for key in unsaved_questions_with_answers["questions"]:
        questions_with_answers["questions"][key] = unsaved_questions_with_answers["questions"][key]
    
    with open(questions_with_answers_path, "w", encoding="utf-8") as questions_file:
        json.dump(questions_with_answers, questions_file)
    # create backup for unsaved_questions_with_answers
    backup_unsaved_questions_with_answers_path = os.path.join(
        settings.BASE_DIR, "static", "chatbot_questions/unsaved_questions_with_answers" +
        datetime.datetime.now().strftime("-%M_%d_%Y-%X")+".json"
    )
    with open(backup_unsaved_questions_with_answers_path, "w+", encoding="utf-8") as backup_unsaved_questions_with_answers_file:
        json.dump(unsaved_questions_with_answers,
                  backup_unsaved_questions_with_answers_file)
    with open(unsaved_questions_with_answers_path, "w", encoding="utf-8") as unsaved_questions_with_answers_file:
        json.dump({"questions": {}}, unsaved_questions_with_answers_file)
