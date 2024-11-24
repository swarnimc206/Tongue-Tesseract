from transformers import MarianMTModel, MarianTokenizer

# Dictionary to hold models and tokenizers
models = {}
tokenizers = {}

# List of model names for each language pair
language_pairs = {
    'en-hi': 'Helsinki-NLP/opus-mt-en-hi',
    'hi-en': 'Helsinki-NLP/opus-mt-hi-en'
}

# Load models and tokenizers
for pair, model_name in language_pairs.items():
    tokenizers[pair] = MarianTokenizer.from_pretrained(model_name)
    models[pair] = MarianMTModel.from_pretrained(model_name)
    
    # Save the model and tokenizer to a directory
    models[pair].save_pretrained(f'./saved_models/{pair}')
    tokenizers[pair].save_pretrained(f'./saved_models/{pair}')

# Define a translation function
def translate(text, pair):
    tokenizer = tokenizers[pair]
    model = models[pair]
    
    # Tokenize the text
    encoded_text = tokenizer.prepare_seq2seq_batch([text], return_tensors='pt')
    
    # Generate translation using the model
    translated = model.generate(**encoded_text)
    
    # Decode the translated text
    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return translated_text[0]
