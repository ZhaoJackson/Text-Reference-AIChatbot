# utils.py
from src.commonconst import *
from transformers import BertTokenizer, TFBertForSequenceClassification
import csv
import nltk
import pandas as pd
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# Initialize BERT model and tokenizer for ethical alignment
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
model = TFBertForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=BERT_NUM_LABELS)

def load_responses(file_path):
    """Loads responses from the given CSV file."""
    responses = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            responses.append({
                'Platform': row['Platform'],
                'Response': row['Response']
            })
    return responses

def calculate_average_rouge(reference_text, generated_text):
    """Calculates the average ROUGE score for the generated text compared to the reference text."""
    scorer = rouge_scorer.RougeScorer(ROUGE_METRICS, use_stemmer=ROUGE_USE_STEMMER)
    scores = scorer.score(reference_text, generated_text)
    avg_rouge = round(sum(scores[metric].fmeasure for metric in ROUGE_METRICS) / len(ROUGE_METRICS), 2)
    return avg_rouge

def calculate_meteor(reference_text, generated_text):
    """Calculates the METEOR score for the generated text compared to the reference text."""
    reference_tokens = nltk.word_tokenize(reference_text.lower())
    hypothesis_tokens = nltk.word_tokenize(generated_text.lower())
    meteor = meteor_score([reference_tokens], hypothesis_tokens)
    return round(meteor, 2)

def evaluate_ethical_alignment(reference_text, generated_text):
    """Evaluates the ethical alignment score of the generated text."""
    inputs = tokenizer(generated_text, return_tensors='tf', truncation=True, padding=True, max_length=MAX_LENGTH)
    outputs = model(inputs)
    scores = outputs.logits[0].numpy()
    ethical_score = scores[1]  # Assuming index 1 corresponds to ethical alignment
    return round(ethical_score, 2)

def evaluate_sentiment_distribution(reference_text, generated_text):
    """Evaluates the sentiment distribution score of the generated text."""
    emotion_analysis = [{'label': 'joy', 'score': 0.8}, {'label': 'sadness', 'score': 0.1}]  # Example placeholder
    emotion_scores = {score['label']: score['score'] for score in emotion_analysis if score['label'] in RELEVANT_EMOTIONS}
    sentiment_score = sum(emotion_scores.values()) / len(emotion_scores) if emotion_scores else 0
    return round(sentiment_score, 2)

def evaluate_inclusivity_score(reference_text, generated_text):
    """Evaluates the inclusivity score of the generated text."""
    words = nltk.word_tokenize(generated_text.lower())
    inclusive_count = sum(1 for word in words if word in INCLUSIVITY_LEXICON)
    inclusivity_score = inclusive_count / len(words) if len(words) > 0 else 0
    return round(inclusivity_score, 2)

def evaluate_complexity_score(reference_text, generated_text):
    """Evaluates the complexity score of the generated text using readability metrics."""
    sentences = nltk.sent_tokenize(generated_text)
    avg_sentence_length = sum(len(nltk.word_tokenize(sentence)) for sentence in sentences) / len(sentences) if sentences else 0
    total_words = sum(len(nltk.word_tokenize(sentence)) for sentence in sentences)
    cmudict = nltk.corpus.cmudict.dict()

    def count_syllables(word):
        phonemes_list = cmudict.get(word.lower(), [[0]])
        return sum(1 for phoneme in phonemes_list[0] if isinstance(phoneme, str) and phoneme[-1].isdigit())

    total_syllables = sum(count_syllables(word) for word in nltk.word_tokenize(generated_text))
    fk_score = READABILITY_FK_CONSTANT - READABILITY_FK_SENTENCE_WEIGHT * (total_words / len(sentences)) - \
               READABILITY_FK_SYLLABLE_WEIGHT * (total_syllables / total_words) if total_words > 0 else 0
    complexity_score = (avg_sentence_length + fk_score) / 2
    return round(complexity_score, 2)

def generate_evaluation_scores(integrated_responses):
    """Generates evaluation scores for each chatbot platform by comparing with the human response."""
    evaluation_data = []

    # Load human reference response for comparison
    human_response = next(item['Response'] for item in integrated_responses if item['Platform'] == 'Human')

    for response in integrated_responses:
        if response['Platform'] == 'Human':
            continue  # Skip the human response for evaluation; only compare chatbots

        generated_text = response['Response']
        avg_rouge = calculate_average_rouge(human_response, generated_text)
        meteor = calculate_meteor(human_response, generated_text)
        ethical_alignment = evaluate_ethical_alignment(human_response, generated_text)
        sentiment_distribution = evaluate_sentiment_distribution(human_response, generated_text)
        inclusivity_score = evaluate_inclusivity_score(human_response, generated_text)
        complexity_score = evaluate_complexity_score(human_response, generated_text)

        evaluation_data.append({
            'Chatbot': response['Platform'],
            'Response': generated_text,
            'Average ROUGE Score': avg_rouge,
            'METEOR Score': meteor,
            'Ethical Alignment Score': ethical_alignment,
            'Sentiment Distribution Score': sentiment_distribution,
            'Inclusivity Score': inclusivity_score,
            'Complexity Score': complexity_score
        })
    return evaluation_data

def save_evaluation_to_csv(file_path, evaluation_data):
    """Saves evaluation data to a CSV file."""
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=EVALUATION_FIELDNAMES)
        writer.writeheader()
        writer.writerows(evaluation_data)