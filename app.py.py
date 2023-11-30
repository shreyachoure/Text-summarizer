from transformers import T5ForConditionalGeneration, T5Tokenizer
import streamlit as st

# Path to your fine-tuned model
model_path = "./Model"
token_path = "./Model"
# Check for GPU availability
device = "cpu"

# Load tokenizer and model

tokenizer = T5Tokenizer.from_pretrained(token_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)


def summarize(conversation):
    inputs = tokenizer.encode(
        "summarize: " + conversation,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).to(device)
    outputs = model.generate(
        inputs,
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


import streamlit as st

# Streamlit app for testing
st.title("Conversation Summarizer")

conversation = st.text_area("Enter the conversation here:")
if st.button("Summarize"):
    summary = summarize(conversation)
    st.write(summary)
