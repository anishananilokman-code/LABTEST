import os
import re
import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
from PyPDF2 import PdfReader
import pandas as pd

st.set_page_config(page_title="Text Chunking with NLTK", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Text Chunking Web App</h1>", unsafe_allow_html=True)
st.caption("Extract and chunk sentences semantically from PDF text using NLTK")

# ----------------------------
# NLTK setup (cached)
# ----------------------------
@st.cache_resource
def setup_nltk():
    nltk_data_path = "/tmp/nltk_data"
    os.makedirs(nltk_data_path, exist_ok=True)
    nltk.data.path.append(nltk_data_path)

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", download_dir=nltk_data_path)

setup_nltk()

# ----------------------------
# Upload PDF
# ----------------------------
st.sidebar.header("Upload Your PDF Document")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    reader = PdfReader(uploaded_file)

    # Extract text
    pages_text = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            pages_text.append(t)

    document_text = "\n".join(pages_text)

    # Basic cleaning (helps sentence tokenizer)
    document_text = re.sub(r"\s+", " ", document_text).strip()

    if not document_text:
        st.error("Could not extract any text from this PDF. It might be an image-only scan.")
    else:
        # Tokenize into sentences (semantic chunks)
        sentences = sent_tokenize(document_text)

        st.subheader("Summary")
        st.write(f"Total sentences extracted: **{len(sentences)}**")

        # Sample 58–68
        st.subheader("Sample Extracted Sentences (Index 58–68)")
        if len(sentences) >= 69:
            sample_df = pd.DataFrame({
                "Sentence Index": list(range(58, 69)),
                "Sentence": sentences[58:69]
            })
            st.dataframe(sample_df, use_container_width=True)
        else:
            st.warning(
                f"The document only contains {len(sentences)} sentences. "
                f"Not enough to show indices 58–68."
            )

        # Full chunk output
        st.subheader("🔍 Semantic Sentence Chunking Output")
        chunk_df = pd.DataFrame({
            "Sentence Index": list(range(len(sentences))),
            "Sentence": sentences
        })
        st.dataframe(chunk_df, use_container_width=True)

        st.info(
            "Each sentence is treated as one semantic chunk using NLTK's sentence tokenizer, "
            "which supports further NLP analysis."
        )

