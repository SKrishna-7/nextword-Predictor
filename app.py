import streamlit as st

import pickle

from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


model=load_model('./saved_models/model.pkl')

with open('./saved_models/tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)


def predict_nextword(model,tokenizer,text,max_sequence_len):
    token_list=tokenizer.texts_to_sequences([text])[0]

    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None


st.title("Next Word Prediction With LSTM ")
input_text=st.text_input("Enter the sequence of Words","To be or not to")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
    next_word = predict_nextword(model, tokenizer, input_text, max_sequence_len)
    st.write(f'Next word: {next_word}')

st.markdown("<p style='font-size:12px; text-align:center;'>Developed by Suresh Krishnan <a href='https://www.linkedin.com/in/suresh-krishnan-s/' target='_blank'>LinkedIn</a> | <a href='https://github.com/SKrishna-7' target='_blank'>GitHub</a></p>", unsafe_allow_html=True )

