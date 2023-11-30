from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
)

import streamlit as st

# Define model paths
model_paths = {
    "T5 Fine Tuned": "./Model/T5",
    "T5 RAW": "./Model/T5 RAW",
    "BART Fine Tuned": "./Model/BART",
    "BART RAW": "./Model/BART RAW",
}

# Streamlit app
st.title("Conversation Summarizer")

# Dropdown to select the model
selected_model = st.selectbox("Select a Model", list(model_paths.keys()))

tokenizer = None
tokenizer = None
if selected_model == "T5 Fine Tuned" or selected_model == "T5 RAW":
    # Load tokenizer and model based on selection
    tokenizer = T5Tokenizer.from_pretrained(model_paths[selected_model])
    model = T5ForConditionalGeneration.from_pretrained(model_paths[selected_model]).to(
        "cpu"
    )
else:
    tokenizer = BartTokenizer.from_pretrained(model_paths[selected_model])
    tokenizer = BartForConditionalGeneration.from_pretrained(
        model_paths[selected_model]
    ).to("cpu")


def summarize(conversation):
    inputs = tokenizer.encode(
        "summarize: " + conversation,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).to("cpu")
    outputs = model.generate(
        inputs,
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Input for conversation
conversation = st.text_area("Enter the conversation here:")
if st.button("Summarize"):
    summary = summarize(conversation)
    st.write(summary)
