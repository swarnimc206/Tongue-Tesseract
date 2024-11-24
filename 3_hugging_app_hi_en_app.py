import streamlit as st

from transformers import MarianMTModel, MarianTokenizer

# Check if models and tokenizers are already loaded to use session state
if 'models' not in st.session_state or 'tokenizers' not in st.session_state:
    st.session_state['models'] = {}
    st.session_state['tokenizers'] = {}

    # List of model names for each language pair
    language_pairs = {
        'en-hi': 'Helsinki-NLP/opus-mt-en-hi',
        'hi-en': 'Helsinki-NLP/opus-mt-hi-en'
    }

    # Load models and tokenizers from the saved_models directory
    for pair in language_pairs:
        st.session_state['tokenizers'][pair] = MarianTokenizer.from_pretrained(f'./saved_models/{pair}')
        st.session_state['models'][pair] = MarianMTModel.from_pretrained(f'./saved_models/{pair}')

def translate(text, pair):
    tokenizer = st.session_state['tokenizers'][pair]
    model = st.session_state['models'][pair]

    # Tokenize the text
    encoded_text = tokenizer.prepare_seq2seq_batch([text], return_tensors='pt')
    
    # Generate translation using the model
    translated = model.generate(**encoded_text)
    
    # Decode the translated text
    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return translated_text[0]

# UI
st.sidebar.title('Tongue Tesseract')
st.sidebar.subheader('Employing NLP for Language Translation')

# Sidebar for selecting language pair
option = st.sidebar.selectbox('Choose translation direction:', ('en-hi', 'hi-en'))

# Text input in main area
st.title('Language Translator')
input_text = st.text_area("Input Text", height=150)

# Translate button in sidebar
if st.sidebar.button('Translate'):
    if input_text:
        translation = translate(input_text, option)
        st.text_area("Translation", translation, height=150, key='output')
    else:
        st.sidebar.warning('Please enter text to translate.')

# Adding some information about the project in the sidebar
st.sidebar.markdown('### Project Information')
st.sidebar.info('This project, "Tongue Tesseract", leverages advanced NLP models to translate text between Hindi and English, demonstrating the power of language translation technologies. Developed using Streamlit and Hugging Face Transformers.')
