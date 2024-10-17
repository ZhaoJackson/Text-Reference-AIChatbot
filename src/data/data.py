from src.commonconst import *

def extract_text_from_docx(doc_path):
    """Extracts text from a .docx file."""
    doc = docx.Document(doc_path)
    text = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip() != ""]
    return text

def process_reference_text(reference_text):
    """Processes the reference text into a structured format for CSV output."""
    data = []
    current_section = None
    for line in reference_text:
        if line.endswith(SECTION_SUFFIX):
            current_section = line[:-1].strip()  # Remove the section suffix
        else:
            data.append({
                "Platform": HUMAN_PLATFORM,
                "Topics": current_section,
                "Response": line
            })
    return data

def process_chatbot_responses(chatbot_text):
    """Processes the chatbot responses into a structured format for CSV output."""
    data = []
    current_chatbot = None
    current_section = None
    for line in chatbot_text:
        if RESPONSE_PREFIX in line:
            current_chatbot = line.split(RESPONSE_PREFIX)[-1].strip()
        elif line.endswith(SECTION_SUFFIX):
            current_section = line[:-1].strip()  # Remove the section suffix
        else:
            data.append({
                "Platform": current_chatbot,
                "Topics": current_section,
                "Response": line
            })
    return data

def save_to_csv(file_path, fieldnames, data):
    """Saves data to a CSV file."""
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)