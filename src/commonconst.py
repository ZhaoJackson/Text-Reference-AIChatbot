# Copyright (c) 2025 Zichen Zhao
# Columbia University School of Social Work
# Licensed under the MIT Academic Research License
# See LICENSE file in the project root for details.

# =================================
# LIBRARY IMPORTS
# =================================
import os
import csv
import docx
import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from transformers import BertTokenizer, TFBertForSequenceClassification, pipeline
from sklearn.metrics.pairwise import cosine_similarity

# ==================================
# FILE PATHS
# ==================================

# Data input paths
REFERENCE_DOCX_PATH = 'src/data/Test Reference Text.docx'
CHATBOT_DOCX_PATH = 'src/data/Test Chatbot text.docx'

# Data output paths
REFERENCE_CSV_PATH = 'src/data/reference_text.csv'
CHATBOT_CSV_PATH = 'src/data/chatbot_text.csv'
OUTPUT_CSV_PATH = 'src/outputs/evaluation_scores.csv'
INTEGRATED_OUTPUT_CSV_PATH = 'src/outputs/integrated_chatbot_responses.csv'
CHATBOT_PROCESSED_CSV_PATH = 'src/outputs/processed_chatbot_text.csv'
REFERENCE_PROCESSED_CSV_PATH = 'src/outputs/processed_reference_text.csv'

# Visualization output path
PLOTS_DIR = 'src/outputs/Plots'

# ==================================
# CSV FIELDNAMES
# ==================================

# Input filde names
FIELDNAMES = ["Platform", "Topics", "Response"]

# Algorithm output field names
EVALUATION_FIELDNAMES = [
    'Chatbot', 'Response', 'Average ROUGE Score', 'METEOR Score',
    'Ethical Alignment Score', 'Sentiment Distribution Score',
    'Inclusivity Score', 'Complexity Score'
]
# Visualization field names
VISUALIZATION_METRICS = [
    'Average ROUGE Score', 'METEOR Score', 'Ethical Alignment Score',
    'Inclusivity Score', 'Sentiment Distribution Score', 'Complexity Score'
]

# ==================================
# GENERAL CONSTANTS for Algorithm Processing
# ==================================
HUMAN_PLATFORM = "Human"
RESPONSE_PREFIX = "Response from"
SECTION_SUFFIX = ':'

# ==================================
# MODEL CONFIGURATIONS
# ==================================

# Ethical Alignment Model
BERT_MODEL_NAME = 'bert-base-uncased'
BERT_NUM_LABELS = 2

# Sentiment Distribution Model
EMOTIONAL_MODEL = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

# ROUGE Score Calculation (Configuration)
ROUGE_METRICS = ['rouge1', 'rouge2', 'rougeL']
ROUGE_USE_STEMMER = True

# ETHICAL SCORING CONSTANTS
MAX_LENGTH = None

# Sentiment Distribution Calculation
RELEVANT_EMOTIONS = [
    'empathy', 'compassion', 'validation', 'understanding', 'trust', 'support',
    'safety', 'reassurance', 'joy', 'love', 'optimism', 'hope', 'relief', 'calm',
    'gratitude', 'caring', 'confident', 'sadness', 'fear', 'anxiety', 'anger',
    'shame', 'guilt', 'loneliness', 'isolation', 'confusion', 'neutral',
    'surprise', 'curiosity'
]

EMOTION_WEIGHTS = {
    'empathy': 2.5, 'compassion': 2.5, 'validation': 2.2, 'understanding': 2.0,
    'trust': 2.0, 'support': 1.8, 'safety': 1.8, 'reassurance': 1.6,
    'joy': 1.4, 'love': 1.6, 'optimism': 1.5, 'hope': 1.6,
    'relief': 1.3, 'calm': 1.2, 'gratitude': 1.2, 'caring': 1.5, 'confident': 1.3,
    'sadness': 0.9, 'fear': 0.8, 'anxiety': 0.8, 'anger': 0.6, 'shame': 0.5,
    'guilt': 0.5, 'loneliness': 0.6, 'isolation': 0.6, 'confusion': 0.6,
    'neutral': 0.4, 'surprise': 0.5, 'curiosity': 0.6
}

# INCLUSIVITY SCORING TERMS
INCLUSIVITY_LEXICON = {
    'inclusive', 'diverse', 'equitable', 'accessibility', 'non-binary',
    'gender nonconforming', 'gender identity', 'sexual orientation',
    'LGBTQ+ support', 'identity acceptance', 'discrimination',
    'safe space', 'affirmation', 'gender-affirming', 'allyship',
    'support system', 'resilience', 'self-worth', 'healing-centered',
    'mental health advocate', 'compassionate support'
}
CORE_TERMS = {
    'gender identity', 'sexual orientation', 'LGBTQ+', 'identity acceptance',
    'safe space', 'allyship', 'inclusive language', 'authentic self'
}
SECONDARY_TERMS = {
    'resilience', 'culturally appropriate', 'psychological safety',
    'connected to community', 'trusted person', 'inclusive provider'
}
PENALTY_TERMS = {
    'crazy', 'normal', 'weak', 'abnormal', 'insane', 'burden', 'failure'
}
SEVERE_PENALTY_TERMS = {
    'psychotic', 'schizo', 'delusional', 'mental case'
}

# READABILITY METRIC (Flesch-Kincaid) CONSTANTS
READABILITY_CONSTANTS = {
    'READABILITY_FK_CONSTANT': 206.835,
    'READABILITY_FK_SENTENCE_WEIGHT': 1.1,
    'READABILITY_FK_SYLLABLE_WEIGHT': 70.0,
    'SENTENCE_COMPLEXITY_WEIGHT': 1.2
}