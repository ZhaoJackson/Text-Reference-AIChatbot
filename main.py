from src.commonconst import *
from src.data.data import *
from src.utils.evaluation_algo import *

def main():
    # Extract and process reference text
    reference_text = extract_text_from_docx(REFERENCE_DOCX_PATH)
    reference_data = process_reference_text(reference_text)
    save_to_csv(REFERENCE_CSV_PATH, FIELDNAMES, reference_data)

    # Extract and process chatbot responses
    chatbot_text = extract_text_from_docx(CHATBOT_DOCX_PATH)
    chatbot_data = process_chatbot_responses(chatbot_text)
    save_to_csv(CHATBOT_CSV_PATH, FIELDNAMES, chatbot_data)

    # Load processed responses for evaluation
    chatbot_responses = load_responses(CHATBOT_CSV_PATH)
    reference_responses = load_responses(REFERENCE_CSV_PATH)

    # Generate evaluation scores
    evaluation_scores = generate_evaluation_scores(chatbot_responses, reference_responses)
    save_evaluation_to_csv(OUTPUT_CSV_PATH, evaluation_scores)

    print("Data processing and evaluation complete. Results saved in the src/outputs folder.")

if __name__ == "__main__":
    main()