# data_processing.py
from src.commonconst import *

def extract_text_from_docx(doc_path):
    """Extracts text from a .docx file and filters out empty paragraphs."""
    doc = docx.Document(doc_path)
    text = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip() != ""]
    return text

def process_reference_text(reference_text):
    """Processes the reference text into a structured format for CSV output."""
    data = []
    current_section = None
    for line in reference_text: # scan each line in the reference text
        if line.endswith(SECTION_SUFFIX): # detects section headers 
            current_section = line[:-1].strip()
        else:
            data.append({
                "Platform": HUMAN_PLATFORM,
                "Topics": current_section,
                "Response": line # stores each response under its corresponding topic section and platform and format as this one
            })
    return data

def process_chatbot_responses(chatbot_text):
    """Processes the chatbot responses into a structured format for CSV output."""
    data = []
    current_chatbot = None
    current_section = None
    for line in chatbot_text:
        if RESPONSE_PREFIX in line: # detects chatbot names
            current_chatbot = line.split(RESPONSE_PREFIX)[-1].strip()
        elif line.endswith(SECTION_SUFFIX): # track both chatbot names and topic sections
            current_section = line[:-1].strip() # store each chatbot-generated sentence under its corresponding topic section
        else:
            data.append({
                "Platform": current_chatbot,
                "Topics": current_section,
                "Response": line
            })
    return data

def save_processed_files(chatbot_text, reference_text, chatbot_output_path, reference_output_path, integrated_output_path):
    """Processes and saves chatbot, reference text, and integrated responses into CSV files."""
    
    def save_to_csv(file_path, fieldnames, data):
        """Saves data to a CSV file."""
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    # Process and save chatbot responses
    chatbot_data = process_chatbot_responses(chatbot_text)
    save_to_csv(chatbot_output_path, ['Platform', 'Topics', 'Response'], chatbot_data)
    
    # Process and save reference text
    reference_data = process_reference_text(reference_text)
    save_to_csv(reference_output_path, ['Platform', 'Topics', 'Response'], reference_data)
    
    # Integrate responses and save the combined result
    chatbot_text_df = pd.DataFrame(chatbot_data)
    reference_text_df = pd.DataFrame(reference_data)
    
    # Aggregating responses in chatbot_text by platform
    chatbot_aggregated = chatbot_text_df.groupby('Platform')['Response'].apply(' '.join).reset_index()
    
    # Aggregating all responses for the Human platform in reference_text
    human_response = ' '.join(reference_text_df['Response'].tolist())
    reference_aggregated = pd.DataFrame([{'Platform': 'Human', 'Response': human_response}])
    
    # Combine chatbot and reference text into one dataframe
    integrated_df = pd.concat([chatbot_aggregated, reference_aggregated], ignore_index=True)
    integrated_df.to_csv(integrated_output_path, index=False)
    
    print(f"Processed chatbot responses saved to {chatbot_output_path}")
    print(f"Processed reference text saved to {reference_output_path}")
    print(f"Integrated responses saved to {integrated_output_path}")