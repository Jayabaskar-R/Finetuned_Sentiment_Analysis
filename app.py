import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Define the local model path
MODEL_PATH = "C:/Fine_Tuning/fine_tuned_sentiment_model"

# Load the model and tokenizer
@st.cache_resource
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Define label mapping
label_mapping = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}

st.title("Twitter Sentiment Analysis")
st.write("Analyze the sentiment of tweets!")

text = st.text_area("Enter a tweet:")

if st.button("Analyze"):
    if text.strip():
        if model:
            with st.spinner("Analyzing..."):
                result = model(text)
                sentiment_label = label_mapping.get(result[0]['label'], "Unknown")  # Map labels
            st.success(f"**Sentiment:** {sentiment_label} (Score: {result[0]['score']:.2f})")
        else:
            st.error("Model failed to load. Check the path and try again.")
    else:
        st.warning("Please enter some text.")
