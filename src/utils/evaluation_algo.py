# utils.py
from src.commonconst import *

# Initialize BERT model and tokenizer for ethical alignment
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
ethical_model = TFBertForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=BERT_NUM_LABELS)

# Initialize Sentiment distribution model
emotion_model = EMOTIONAL_MODEL

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
    """Calculates an adjusted average ROUGE score with tuned weights for precision and recall, favoring compassionate and inclusive language."""
    scorer = rouge_scorer.RougeScorer(ROUGE_METRICS, use_stemmer=ROUGE_USE_STEMMER)
    scores = scorer.score(reference_text, generated_text)
    avg_rouge = round(
        sum(
            (scores['rouge1'].precision * 0.5 + scores['rouge1'].recall * 0.5) * 0.4 +
            (scores['rouge2'].precision * 0.6 + scores['rouge2'].recall * 0.4) * 0.3 +
            (scores['rougeL'].precision * 0.4 + scores['rougeL'].recall * 0.6) * 0.3
            for metric in ROUGE_METRICS
        ) / len(ROUGE_METRICS), 2
    )
    return avg_rouge

def calculate_meteor(reference_text, generated_text):
    """Calculates the METEOR score with tuned parameters for increased synonym matching and improved recall balance."""
    reference_tokens = nltk.word_tokenize(reference_text.lower())
    hypothesis_tokens = nltk.word_tokenize(generated_text.lower())
    meteor = meteor_score([reference_tokens], hypothesis_tokens, alpha=0.8, beta=1.5, gamma=0.6)
    return round(meteor, 2)

def evaluate_ethical_alignment(reference_text, generated_text):
    """Simplified ethical alignment for binary classifier. Focuses on ethical appropriateness for LGBTQ+ mental health."""
    inputs = tokenizer(generated_text, return_tensors='tf', truncation=True, padding=True, max_length=MAX_LENGTH)
    outputs = ethical_model(inputs)
    probs = tf.nn.softmax(outputs.logits, axis=1)[0].numpy()
    
    # Use probability of the "ethically aligned" class (index 1)
    ethical_score = float(probs[1])
    
    # Weighting scheme aligned with thesis (context-sensitive amplification)
    if ethical_score > 0.8:
        weighted_score = ethical_score
    elif ethical_score > 0.6:
        weighted_score = ethical_score * 0.98
    elif ethical_score > 0.4:
        weighted_score = ethical_score * 0.9
    else:
        weighted_score = ethical_score * 0.5 

    return round(weighted_score, 2)

def evaluate_sentiment_distribution(reference_text, generated_text, emotion_weights):  
    def get_weighted_vector(text):
        raw_emotions = emotion_model(text)[0]
        emotion_dict = {e['label'].lower(): e['score'] for e in raw_emotions}
        return np.array([
            emotion_dict.get(emotion, 0.0) * emotion_weights.get(emotion, 1.0)
            for emotion in RELEVANT_EMOTIONS
            ]).reshape(1, -1)
    ref_vec = get_weighted_vector(reference_text)
    gen_vec = get_weighted_vector(generated_text)
    similarity = cosine_similarity(ref_vec, gen_vec)[0][0]
    return round(float(similarity), 2)

def evaluate_inclusivity_score(reference_text, generated_text):
    """Evaluates the inclusivity score of the generated text, emphasizing affirming language and penalizing non-inclusive terms. Scores are scaled between 0 and 1."""
    words = nltk.word_tokenize(generated_text.lower())
    inclusive_count = sum(
        4 if word in CORE_TERMS else 2.5 if word in SECONDARY_TERMS else 2
        for word in words if word in INCLUSIVITY_LEXICON
    )
    penalty_count = sum(
        1.0 if word in SEVERE_PENALTY_TERMS else 0.5 
        for word in words if word in PENALTY_TERMS
    )
    total_words = len(words)
    inclusivity_density = (inclusive_count - penalty_count) / total_words if total_words > 0 else 0
    inclusivity_score = max(0, round(inclusivity_density + (inclusive_count / 15), 2))
    return inclusivity_score

def evaluate_complexity_score(reference_text, generated_text, readability_constants):
    """Evaluates the complexity score of the generated text, prioritizing balanced readability and accessibility for mental health contexts."""
    sentences = nltk.sent_tokenize(generated_text)
    avg_sentence_length = sum(len(nltk.word_tokenize(sentence)) for sentence in sentences) / len(sentences) if sentences else 0
    total_words = sum(len(nltk.word_tokenize(sentence)) for sentence in sentences)
    cmudict = nltk.corpus.cmudict.dict()
    def count_syllables(word):
        phonemes_list = cmudict.get(word.lower(), [[0]])
        return sum(1 for phoneme in phonemes_list[0] if isinstance(phoneme, str) and phoneme[-1].isdigit())
    total_syllables = sum(count_syllables(word) for word in nltk.word_tokenize(generated_text))
    fk_score = (
        readability_constants['READABILITY_FK_CONSTANT'] -
        readability_constants['READABILITY_FK_SENTENCE_WEIGHT'] * (total_words / len(sentences)) -
        readability_constants['READABILITY_FK_SYLLABLE_WEIGHT'] * (total_syllables / total_words)
    ) if total_words > 0 else 0
    complexity_score = (avg_sentence_length * readability_constants['SENTENCE_COMPLEXITY_WEIGHT'] + fk_score) / 2
    return round(complexity_score, 2)

def generate_evaluation_scores(integrated_responses):
    """Generates evaluation scores for each chatbot platform by comparing with the human response."""
    evaluation_data = []
    human_response = next(item['Response'] for item in integrated_responses if item['Platform'] == 'Human')
    
    for response in integrated_responses:
        if response['Platform'] == 'Human':
            continue

        generated_text = response['Response']
        avg_rouge = calculate_average_rouge(human_response, generated_text)
        meteor = calculate_meteor(human_response, generated_text)
        ethical_alignment = evaluate_ethical_alignment(human_response, generated_text)
        sentiment_distribution = evaluate_sentiment_distribution(
            human_response,
            generated_text,
            EMOTION_WEIGHTS
        )
        inclusivity_score = evaluate_inclusivity_score(human_response, generated_text)
        complexity_score = evaluate_complexity_score(human_response, generated_text, READABILITY_CONSTANTS)

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