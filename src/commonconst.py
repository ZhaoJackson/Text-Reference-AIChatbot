import csv
import docx

# File paths
REFERENCE_DOCX_PATH = 'src/data/Test Reference Text.docx'
CHATBOT_DOCX_PATH = 'src/data/Test Chatbot text.docx'
REFERENCE_CSV_PATH = 'src/data/reference_text.csv'
CHATBOT_CSV_PATH = 'src/data/chatbot_text.csv'

# Field names for CSV
FIELDNAMES = ["Platform", "Topics", "Response"]

# Other constants
HUMAN_PLATFORM = "Human"
RESPONSE_PREFIX = "Response from"
SECTION_SUFFIX = ':'