from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
)
from rouge_score import rouge_scorer
import streamlit as st
import pandas as pd

# Define model paths and descriptions
model_paths = {
    "T5 Fine Tuned": "./Model/T5",
    "T5 RAW": "./Model/T5 RAW",
    "BART Fine Tuned": "./Model/BART",
    "BART RAW": "./Model/BART RAW",
}

example_conversations = [
    {
        "conversation": """#Person1#: Hello, how are you doing today?
#Person2#: I ' Ve been having trouble breathing lately.
#Person1#: Have you had any type of cold lately?
#Person2#: No, I haven ' t had a cold. I just have a heavy feeling in my chest when I try to breathe.
#Person1#: Do you have any allergies that you know of?
#Person2#: No, I don ' t have any allergies that I know of.
#Person1#: Does this happen all the time or mostly when you are active?
#Person2#: It happens a lot when I work out.
#Person1#: I am going to send you to a pulmonary specialist who can run tests on you for asthma.
#Person2#: Thank you for your help, doctor.""",
        "summary": "#Person2# has trouble breathing. The doctor asks #Person2# about it and will send #Person2# to a pulmonary specialist.",
    },
    {
        "conversation": """#Person1#: Where are you going for your trip?
#Person2#: I think Hebei is a good place.
#Person1#: But I heard the north of China are experiencing severe sandstorms!
#Person2#: Really?
#Person1#: Yes, it's said that Hebes was experiencing six degree strong winds.
#Person2#: How do these storms affect the people who live in these areas?
#Person1#: The report said the number of people with respiratory tract infections tended to rise after sandstorms. The sand gets into people's noses and throats and creates irritation.
#Person2#: It sounds that sandstorms are trouble for everybody!
#Person1#: You are quite right.""",
        "summary": "#Person2# plans to have a trip in Hebei but #Person1# says there are sandstorms in there.",
    },
    {
        "conversation": """#Person1#: What time is it, Tom?
#Person2#: Just a minute. It's ten to nine by my watch.
#Person1#: Is it? I had no idea it was so late. I must be off now.
#Person2#: What's the hurry?
#Person1#: I must catch the nine-thirty train.
#Person2#: You've plenty of time yet. The railway station is very close. It won't take more than twenty minutes to get there.""",
        "summary": "#Person1# is catching a train. Tom asks #Person1# not to hurry.",
    },
]


model_descriptions = {
    "T5 Fine Tuned": "T5 Fine Tuned is trained on DialogSum Dataset to get more accurate generated summary",
    "T5 RAW": "Original Version of T5 Model",
    "BART Fine Tuned": "BART Fine Tuned is trained on DialogSum Dataset to get more accurate generated summary",
    "BART RAW": "Original Version of BART RAW",
}

# Streamlit app setup
st.title("Conversation Summarizer")

# Sidebar for model selection
st.sidebar.title("Model Selection")
selected_model = st.sidebar.selectbox("Choose a Model", list(model_paths.keys()))

# Display model information
st.sidebar.title("Model Information")
st.sidebar.write(model_descriptions[selected_model])


# Example data for average ROGUE scores
data = {
    "Models": ["T5 Fine Tuned", "T5 RAW", "BART Fine Tuned", "BART RAW"],
    "Average Scores": [0.39, 0.18, 0.40, 0.19],  # Example scores
}

# Convert data into a Pandas DataFrame
df = pd.DataFrame(data)

# Set the index to 'Models' for better chart labeling
df = df.set_index("Models")

# Create a bar chart
st.sidebar.title("Average ROGUE Scores")
st.sidebar.bar_chart(df)


# Load tokenizer and model based on selection
if "T5" in selected_model:
    tokenizer = T5Tokenizer.from_pretrained(model_paths[selected_model])
    model = T5ForConditionalGeneration.from_pretrained(model_paths[selected_model]).to(
        "cpu"
    )
else:
    tokenizer = BartTokenizer.from_pretrained(model_paths[selected_model])
    model = BartForConditionalGeneration.from_pretrained(
        model_paths[selected_model]
    ).to("cpu")


# Function to summarize conversation
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


# Function to calculate ROUGE score
def calculate_rouge_score(summary, reference):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return scores


# Main area for user input and output
conversation = st.text_area("Enter the conversation here:")
reference_summary = st.text_area("Enter reference summary (optional):")

if st.button("Summarize"):
    summary = summarize(conversation)
    st.write("Generated Summary:", summary)

    # Calculate and display ROUGE score if reference summary is provided
    if reference_summary:
        rouge_scores = calculate_rouge_score(summary, reference_summary)
        st.write("ROUGE Scores:", rouge_scores)

with st.expander("See Example Conversations and Summaries"):
    st.write(
        "Below are some example conversations with their reference summaries. You can copy these into the input fields above to see how the summarizer works."
    )
    for i, example in enumerate(example_conversations, start=1):
        st.subheader(f"Example {i}")
        st.text_area(
            "Conversation", example["conversation"], key=f"conv{i}", height=100
        )
        st.text_area(
            "Reference Summary", example["summary"], key=f"summary{i}", height=50
        )

with st.expander("Training Dataset"):
    # Dataset Information
    st.markdown(
        """
    **Dataset Information:**
    This summarizer is trained on a specialized dataset designed for conversation summarization. For more details about the dataset and its structure, you can visit the following link: [Dataset Details](https://huggingface.co/datasets/knkarthick/dialogsum/tree/main).
    """
    )
# Expandable section for model limitations
with st.expander("Model Limitations"):
    st.markdown(
        """
    **1. Limited Training of the Models:** 
    The models are trained on a specific set of data, which may not cover all types of conversations. This limitation can affect the accuracy and relevance of the generated summaries.

    **2. Input and Output Size Constraints:** 
    - **Minimum Input Requirement:** Each conversation input must contain at least 60 characters to ensure sufficient context for summarization.
    - **Maximum Output Size:** The output size is limited to ensure concise summaries and efficient processing. This means some details from the original conversation might be omitted in the summary.
    """
    )
