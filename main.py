from src.commonconst import *
from src.data.data_processing import *
from src.utils.evaluation_algo import *
from src.outputs.output_processing import *

def main():
    # Step 1: Extract and process reference text
    reference_text = extract_text_from_docx(REFERENCE_DOCX_PATH)
    reference_data = process_reference_text(reference_text)
    
    # Step 2: Extract and process chatbot responses
    chatbot_text = extract_text_from_docx(CHATBOT_DOCX_PATH)
    chatbot_data = process_chatbot_responses(chatbot_text)
    
    # Step 3: Generate and save processed files, including integrated responses
    save_processed_files(
        chatbot_text=chatbot_text,
        reference_text=reference_text,
        chatbot_output_path=CHATBOT_PROCESSED_CSV_PATH,
        reference_output_path=REFERENCE_PROCESSED_CSV_PATH,
        integrated_output_path=INTEGRATED_OUTPUT_CSV_PATH
    )

    # Step 4: Load integrated responses for evaluation
    integrated_responses = load_responses(INTEGRATED_OUTPUT_CSV_PATH)

    # Step 5: Generate evaluation scores by comparing each chatbot to the human reference
    evaluation_scores = generate_evaluation_scores(integrated_responses)
    save_evaluation_to_csv(OUTPUT_CSV_PATH, evaluation_scores)
    print("Data processing and evaluation complete. Results saved in the src/outputs folder.")

    # step 6: Generate Visualization Plots
    generate_plots()
    print("All visualizations saved in 'src/outputs/Plots/'")

if __name__ == "__main__":
    main()