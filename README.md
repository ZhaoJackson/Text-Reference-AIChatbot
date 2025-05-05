# Enhancing Equity and Actionability in Artificial Intelligence (AI) and Large Language Model (LLM) Chatbots: Machine-Driven Benchmarking for Suicide Prevention Among LGBTQ+ Populations

## Overview
- This project examines the effectiveness of AI chatbots in responding to high-stakes mental health topics, focusing on suicidality within LGBTQ+ communities. By comparing AI-generated responses to expert-crafted human references, this study assesses the AI’s alignment with human standards on precision, ethical alignment, inclusivity, and complexity. The goal is to ensure that chatbots provide supportive, unbiased, and ethically sound assistance, particularly in sensitive mental health contexts.

## Motivation

With mental health chatbots increasingly being used in healthcare, it is vital that they respond with sensitivity, particularly toward vulnerable populations like LGBTQ+ individuals. This project evaluates AI responses in critical mental health scenarios to identify areas where AI responsiveness and empathy can improve. This evaluation highlights gaps in chatbot response quality to foster advancements in AI support for LGBTQ+ mental health.

## Structure
```
Text-Reference-AIChatbot/
├── main.py
├── requirements.txt
├── src/
|   ├── data/
|   |   ├── data_processing.py
|   |   ├── Test Reference Text.docx
|   |   ├── Test Chatbot text.docx
|   ├── utils/
|   |   ├── evaluation_algo.py
|   ├── outputs/
|   |   ├── processed_chatbot_text.csv
|   |   ├── processed_reference_text.csv
|   |   ├── evaluation_scores.csv
|   |   ├── integrated_chatbot_responses.csv
|   |   ├──Plots/
|   └── commonconst.py
└── README.md
```

## Methodology

### 1. Data Preprocessing
- **`data_processing.py`**: Extracts structured data from `.docx` files and converts chatbot/human responses into clean CSV format for analysis.

### 2. Evaluation Metrics (in `evaluation_algo.py`)
Each chatbot response is scored based on the following:

| Metric                     | Output Range | Function | Purpose |
|---------------------------|--------------|----------|---------|
| **ROUGE Score**           | 0–1          | `calculate_average_rouge()` | Measures lexical overlap (precision + recall) between chatbot and human responses. |
| **METEOR Score**          | 0–1          | `calculate_meteor()`        | Accounts for synonyms and stemming; balances exact matches with semantic similarity. |
| **Ethical Alignment**     | 0–1          | `evaluate_ethical_alignment()` | Uses a fine-tuned BERT model to score how ethically appropriate the response is. Applies weighted scaling for moderate confidence cases. |
| **Sentiment Distribution**| 0–1          | `evaluate_sentiment_distribution()` | Compares emotion vectors of chatbot vs. reference using cosine similarity, weighted by therapeutic importance. |
| **Inclusivity Score**     | 0–1          | `evaluate_inclusivity_score()` | Rewards use of affirming, LGBTQ+-inclusive language and penalizes harmful terms. |
| **Complexity Score**      | ~40–70       | `evaluate_complexity_score()` | Combines average sentence length with Flesch-Kincaid readability to assess clarity and depth. |

## Execution Flow
Run `main.py` to:
- Preprocess data
- Apply each metric function
- Save scores to `evaluation_scores.csv`

Each evaluation function returns a numerical score which is logged and compared across chatbots.

---
## Chatbots Evaluated

### 📌 General-Purpose LLMs:
- **ChatGPT-4**
- **Claude (Anthropic)**
- **Gemini (Google)**
- **LLaMA-3 (Meta)**
- **DeepSeek**
- **Mistral**
- **Perplexity AI**
- **HuggingChat**

### 🏳️‍🌈 LGBTQ+-Specific Chatbots:
- **JackAI**
- **Gender Journey Chatbot Rubies**

These platforms were selected for their relevance in AI ethics, mental health, and LGBTQ+ inclusivity—ensuring both high-tech LLMs and community-centric tools are evaluated under equal standards.

---

## Results Summary

### Key Metric Ranges:
- **ROUGE / METEOR**: 0.20–0.36 → Moderate text similarity to reference.
- **Ethical Alignment**: 0.19–0.38 → Indicates general caution but varied empathy.
- **Inclusivity**: 0.00–0.42 → Gaps in affirming language, even in well-performing LLMs.
- **Sentiment Distribution**: 0.04–1.00 → Shows diverse emotional intelligence levels.
- **Complexity**: ~49–61 → Balanced readability with emotional nuance.

### Observations:
- DeepSeek and Gemini lead in inclusivity and sentiment tone.
- ChatGPT-4 and Claude maintain structured inquiry but need ethical fine-tuning.
- Gender Journey performs well on empathy but lags in linguistic richness.

---

## Installation

1. Clone the repository:
```
git clone https://github.com/your-repository
cd your-repository
```

2.	Install dependencies:
```
pip install -r requirements.txt
```

3.	Ensure Python 3.9 or above is installed.

The script will output an evaluation report with scores for each metric, saved in evaluation_scores.csv.

---

## Results Interpretation

| Metric | Insight |
|--------|---------|
| **ROUGE / METEOR** | High = better alignment with human phrasing. |
| **Ethical Alignment** | High = more safety-conscious, affirming language. |
| **Inclusivity** | High = uses LGBTQ+-affirming terms, avoids harm. |
| **Sentiment** | High = tone matches supportive reference. |
| **Complexity** | Mid-range ideal; too low = vague, too high = overly complex. |

---

## Future Work

- **Enhanced Inclusivity**: Further tuning to improve inclusivity specifically for LGBTQ+ language and cultural sensitivity.

- **Ethics Integration**: Implement ethics-driven prompt engineering to enhance alignment with mental health ethics.

- **Dynamic Sentiment Adjustment**: Fine-tune sentiment modulation to adapt responses to specific user needs in real-time.

--- 

For contributions or collaborations, contact **Jackson Zhao** (zz3119@columbia.edu), **Elwin Wu** (elwin.wu@columbia.edu), **Charles Lea** (chl2159@columbia.edu) at **Columbia School of Social Work**