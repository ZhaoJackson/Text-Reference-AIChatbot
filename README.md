# AI Machine-Driven Evaluation of Chatbot Responses in Suicidality Assessment Among LGBTQ+ Individuals

## Overview
- This project evaluates AI chatbot responses in assessing suicidality within the LGBTQ+ community by comparing AI-generated outputs with human reference responses. By measuring precision, ethical alignment, inclusivity, and linguistic complexity, the project aims to ensure chatbot responses are supportive, unbiased, and ethically aligned with human standards in sensitive mental health contexts.

## Motivation

Mental health chatbots are becoming a staple in supportive roles within healthcare. However, ensuring these tools respond with sensitivity and inclusivity, particularly for vulnerable groups like LGBTQ+ individuals, is crucial. This project evaluates chatbot responses in high-stakes scenarios, focusing on suicidality risk assessment, and compares them to human references for improvement in AI responsiveness and empathy.

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
1. Data Preprocessing:
- The chatbot and human response data are preprocessed using data.py to ensure alignment in format for accurate comparisons.

2. Evaluation Metrics:
- Precision (ROUGE, METEOR): Measures textual similarity between chatbot responses and human references.
- Ethical Alignment: Assesses the ethical language and appropriateness of chatbot responses in sensitive contexts.
- Inclusivity Score: Evaluates language for inclusivity, especially relating to LGBTQ+ identity and related nuances.
- Sentiment Distribution: Analyzes response tone to ensure it is appropriate for mental health support.
- Complexity Score: Provides insights into the linguistic sophistication and readability of responses, ensuring clarity without compromising sensitivity.

3. Execution Flow:
- The main.py script orchestrates data processing, metric evaluation using evaluation_algo.py, and result aggregation in evaluation_scores.csv for comparative analysis.

## Results Summary
### Key Findings

The evaluation scores reveal the following trends across chatbots (ChatGPT-4.0, Claude, JackAI):
- ##Precision (ROUGE and METEOR Scores)##: Chatbots show moderate alignment, with ROUGE scores averaging between 0.31 and 0.34 and METEOR scores around 0.27-0.36. This indicates that while chatbot responses are similar to reference texts, there is room for improved textual alignment.

- ##Ethical Alignment##: Ethical scores are relatively low (average of 0.10 to 0.14), highlighting the need for greater adherence to ethical standards in responses related to suicidality.

- ##Inclusivity##: Inclusivity scores are uniformly low, suggesting a gap in language adaptation specifically for LGBTQ+ inclusivity. Further tuning of chatbots is essential for better inclusivity.

- ##Sentiment Distribution and Complexity##: The sentiment distribution remains steady across responses, but complexity scores vary slightly (49.99 to 52.14), suggesting slight differences in readability and linguistic sophistication across models.

The evaluation results suggest that while current AI models perform adequately, improvements are needed in ethical alignment and inclusivity to better serve LGBTQ+ individuals in sensitive mental health contexts.

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

- ROUGE & METEOR Scores: Higher scores indicate better precision and alignment with human responses.

- Ethical & Inclusivity Scores: Scores close to 1 suggest ethical language and inclusivity. Current low scores signal a need for refinement.

- Complexity: Consistent scores indicate similar readability levels across models.

## Future Work

- Enhanced Inclusivity: Further tuning to improve inclusivity specifically for LGBTQ+ language and cultural sensitivity.

- Ethics Integration: Implement ethics-driven prompt engineering to enhance alignment with mental health ethics.

- Dynamic Sentiment Adjustment: Fine-tune sentiment modulation to adapt responses to specific user needs in real-time.