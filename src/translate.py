import tokenize_ro
from transformers import MarianMTModel, MarianTokenizer

if __name__ == '__main__':
    # Load the pre-trained translation model and tokenizer
    # inputs = tokenize_ro.tokenize_file()
    inputs = tokenize_ro.tokenize_file(docx_path="/Users/andreiaalexa/thesis-translation-project/original_thesis_ro1.docx")

    model_name = "Helsinki-NLP/opus-mt-roa-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Translate the text using the model
    translated_tokens = model.generate(**inputs)

    # Detokenize the translated tokens back to text
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    print(translated_text)
