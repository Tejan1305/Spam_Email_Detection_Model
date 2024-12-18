import streamlit as st
import pickle
import string
import re
import spacy
import time

# Page configuration
st.set_page_config(page_title="Spam Email Detector", page_icon="📧", layout="wide")

# Load model and vectorizer
@st.cache_resource(ttl=3600)
def load_model():
    with open('pickle_files\spam_detector.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('pickle_files\count_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# Text cleaning functions
def clean_text(s):
    for cs in s:
        if not cs in string.ascii_letters:
            s = s.replace(cs, ' ')
    return s.rstrip('\r\n')

def remove_little(s):
    words_list = s.split()
    k_length = 2
    result_list = [element for element in words_list if len(element) > k_length]
    result_string = ' '.join(result_list)
    return result_string

def lemmatize_text(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc])
    return lemmatized_text

def preprocess(text):
    return lemmatize_text(remove_little(clean_text(text)))

# Email classification function
def classify_email(model, vectorizer, email):
    prediction = model.predict(vectorizer.transform([email]))
    return prediction

# Main function
def main():
    # Custom CSS for enhanced design
    st.markdown("""
        <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f4f4f4;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .status-bar {
            background-color: #e0e0e0;
            padding: 15px;
            margin-top: 10px;
            border-radius: 5px;
            font-size: 14px;
        }
        .title {
            text-align: center;
            color: #4CAF50;
        }
        </style>
    """, unsafe_allow_html=True)

    # Page layout
    st.markdown("<h1 class='title'>Spam Email Detector 📧</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #555;'>Analyze your email content to detect spam instantly.</p>", unsafe_allow_html=True)

    # Input section
    st.markdown("---")
    st.markdown("### 📝 Enter the email text:")
    user_input = st.text_area(
        label="",
        placeholder="e.g., Congratulations! You have won $1,000,000! Click here to claim your prize!",
        height=200
    )
    st.markdown("---")

    # Button for checking spam
    if st.button("Check for Spam 🚀"):
        if user_input.strip():
            output_placeholder = st.empty()
            status_placeholder = st.empty()

            with status_placeholder.container():
                st.markdown("<div class='status-bar'>Loading the model...</div>", unsafe_allow_html=True)
                time.sleep(1)
                model, vectorizer = load_model()

                st.markdown("<div class='status-bar'>Preprocessing the email content...</div>", unsafe_allow_html=True)
                processed_input = preprocess(user_input)
                time.sleep(1)

                st.markdown("<div class='status-bar'>Analyzing for spam detection...</div>", unsafe_allow_html=True)
                prediction = classify_email(model, vectorizer, processed_input)
                time.sleep(1)

                st.markdown("<div class='status-bar'>Detection completed!</div>", unsafe_allow_html=True)
                time.sleep(0.5)

            # Display results
            status_placeholder.empty()
            if prediction == 1:
                output_placeholder.error("🚨 **Spam Detected!** This email might be harmful or unwanted.")
            else:
                output_placeholder.success("✅ **Not Spam!** This email seems safe.")
        else:
            st.warning("⚠️ Please enter some text to analyze.")

# Run the app
if __name__ == "__main__":
    main()
