from src.commonconst import *
from src.data.data import *

def main():
    # Extract and process reference text
    reference_text = extract_text_from_docx(REFERENCE_DOCX_PATH)
    reference_data = process_reference_text(reference_text)
    save_to_csv(REFERENCE_CSV_PATH, FIELDNAMES, reference_data)

    # Extract and process chatbot responses
    chatbot_text = extract_text_from_docx(CHATBOT_DOCX_PATH)
    chatbot_data = process_chatbot_responses(chatbot_text)
    save_to_csv(CHATBOT_CSV_PATH, FIELDNAMES, chatbot_data)

    print("Data processing complete. Processed CSV files have been generated.")

if __name__ == "__main__":
    main()