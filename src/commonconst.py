# commonconst.py
import csv
import docx
import nltk
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

# Original file paths
REFERENCE_DOCX_PATH = 'src/data/Test Reference Text.docx'
CHATBOT_DOCX_PATH = 'src/data/Test Chatbot text.docx'

# Updated file paths for processed and integrated CSV outputs
REFERENCE_CSV_PATH = 'src/data/reference_text.csv'
CHATBOT_CSV_PATH = 'src/data/chatbot_text.csv'
OUTPUT_CSV_PATH = 'src/outputs/evaluation_scores.csv'
INTEGRATED_OUTPUT_CSV_PATH = 'src/outputs/integrated_chatbot_responses.csv'
CHATBOT_PROCESSED_CSV_PATH = 'src/outputs/processed_chatbot_text.csv'
REFERENCE_PROCESSED_CSV_PATH = 'src/outputs/processed_reference_text.csv'

# Field names for CSV
FIELDNAMES = ["Platform", "Topics", "Response"]

# Evaluation constants
EVALUATION_FIELDNAMES = [
    'Chatbot', 'Response', 'Average ROUGE Score', 'METEOR Score',
    'Ethical Alignment Score', 'Sentiment Distribution Score',
    'Inclusivity Score', 'Complexity Score'
]

# Other constants
HUMAN_PLATFORM = "Human"
RESPONSE_PREFIX = "Response from"
SECTION_SUFFIX = ':'

# Additional constants for evaluation
INCLUSIVITY_LEXICON = {'inclusive', 'diverse', 'equitable', 'accessibility', 'non-binary'}
BERT_MODEL_NAME = 'bert-base-uncased'
BERT_NUM_LABELS = 2
ROUGE_METRICS = ['rouge1', 'rouge2', 'rougeL']
ROUGE_USE_STEMMER = True
MAX_LENGTH = 512
RELEVANT_EMOTIONS = ['joy', 'sadness', 'anger', 'fear', 'trust', 'surprise']
READABILITY_FK_CONSTANT = 206.835
READABILITY_FK_SENTENCE_WEIGHT = 1.015
READABILITY_FK_SYLLABLE_WEIGHT = 84.6