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
|   |   ├── data.py
|   |   ├── Test Reference Text.docx
|   |   ├── Test Chatbot text.docx
|   ├── utils/
|   |   ├── evaluation_algo.py
|   ├── output/
|   |   ├── processed_chatbot_text.csv
|   |   ├── processed_reference_text.csv
|   |   ├── evaluation_scores.csv
|   |   ├── integrated_chatbot_responses.csv
|   └── commonconst.py
└── README.md
```

## Methodology
1. **Data Preprocessing**:
- The `data.py` script standardizes chatbot and human response data formats to facilitate accurate metric comparisons.

2. **Evaluation Metrics**:
- Precision (ROUGE, METEOR): Measures textual similarity between chatbot responses and human references.
- Ethical Alignment: Assesses the ethical language and appropriateness of chatbot responses in sensitive contexts.
- Inclusivity Score: Evaluates language for inclusivity, especially relating to LGBTQ+ identity and related nuances.
- Sentiment Distribution: Analyzes response tone to ensure it is appropriate for mental health support.
- Complexity Score: Provides insights into the linguistic sophistication and readability of responses, ensuring clarity without compromising sensitivity.

3. **Execution Flow**:
- The `main.py` script orchestrates data processing, metric evaluation using evaluation_algo.py, and result aggregation in `evaluation_scores.csv` for comparative analysis.

## Results Summary
### Key Findings

Evaluating multiple chatbots (e.g., ChatGPT-4, Claude, JackAI) revealed the following:
1. **Precision**: ROUGE and METEOR scores averaged between 0.31–0.34 and 0.27–0.36, respectively, indicating moderate alignment with reference texts.
2. **Ethical Alignment**: Ethical scores averaged 0.10–0.14, signaling a need for enhanced adherence to mental health ethics in AI responses.
3. **Inclusivity**: Low inclusivity scores highlighted significant room for improvement in language that is sensitive and supportive toward LGBTQ+ identities.
4. **Sentiment Distribution & Complexity**: Sentiment remained consistent, but complexity scores (49.99–52.14) suggested slight variability in readability, with some responses being either too simplistic or overly complex.

Overall, the results underscore the need for improvements in ethical language alignment and inclusivity to better support LGBTQ+ individuals in mental health contexts.

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

## Results Interpretation

1. **ROUGE & METEOR Scores**: Higher values indicate better alignment with human responses, with room for improved textual matching.
2. **Ethical & Inclusivity Scores**: Scores close to 1 reflect ethical language and inclusivity. Current low scores indicate areas for refinement.
3. **Complexity**: Consistent scores suggest similar readability across models, with adjustments needed for more balanced responses.

## Future Work

- **Enhanced Inclusivity**: Further tuning to improve inclusivity specifically for LGBTQ+ language and cultural sensitivity.

- **Ethics Integration**: Implement ethics-driven prompt engineering to enhance alignment with mental health ethics.

- **Dynamic Sentiment Adjustment**: Fine-tune sentiment modulation to adapt responses to specific user needs in real-time.