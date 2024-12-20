import re
from docx import Document
from transformers import MarianTokenizer


# Step 1: Extract text from .docx file
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"  # Add a newline after each paragraph
    print(f'\n\nExtracted text: {text}')
    return text


# Step 2: Clean the extracted text
def clean_text(text):
    # Remove extra spaces, newlines, tabs
    text = re.sub(r'\s+', ' ', text)

    # Optionally, remove headers/footers or page numbers (customize as needed)
    text = re.sub(r'(Page \d+|Chapter \d+|[A-Za-z]+\s?\d+)', '', text)

    # Remove non-textual elements (e.g., special characters)
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Removes non-ASCII characters

    # Remove redundant punctuation (optional)
    text = re.sub(r'[^\w\s,.!?-]', '', text)  # Keeps only words and common punctuation

    # Optional: Lowercase text (depending on your model's preference)
    # text = text.lower()

    return text.strip()

# Step 3: Tokenize the cleaned text
def tokenize_text(text, model_name="Helsinki-NLP/opus-mt-roa-en"):
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    # Tokenize the text for translation
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    return inputs


# Main function to extract, clean, and tokenize the text
def preprocess_text_for_translation(docx_path):
    # Step 1: Extract text from the .docx file
    extracted_text = extract_text_from_docx(docx_path)
    print("Text extracted from .docx")

    # Step 2: Clean the text
    cleaned_text = clean_text(extracted_text)
    print("Text cleaned")

    # Step 3: Tokenize the cleaned text
    tokenized_inputs = tokenize_text(cleaned_text)
    print("Text tokenized")

    # You can return tokenized_inputs or further process them for translation
    print(f'\n\nTokenized text: {tokenized_inputs}')
    return tokenized_inputs

def tokenize_file(docx_path="/Users/andreiaalexa/thesis-translation-project/original_thesis_ro.docx"):
    return preprocess_text_for_translation(docx_path)

# Example usage:
# if __name__ == "__main__":
#     docx_path = "/Users/andreiaalexa/thesis-translation-project/original_thesis_ro.docx"
#     tokenized_data = preprocess_text_for_translation(docx_path)
#
#     # Optionally, you can print or save the tokenized data
#     # print(f'Tokenized text: {tokenized_data}')
