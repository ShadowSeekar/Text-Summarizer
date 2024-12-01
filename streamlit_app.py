import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch
import base64

# Load model and tokenizer
checkpoint = "MBZUAI/LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(
    checkpoint, device_map="auto", torch_dtype=torch.float32
)

# Function to preprocess the PDF
def file_preprocessing(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = "".join([text.page_content for text in texts])
    return final_texts

# Summarization pipeline
def llm_pipeline(input_text):
    pipe_sum = pipeline(
        "summarization",
        model=base_model,
        tokenizer=tokenizer,
        max_length=500,
        min_length=50,
    )
    results = []
    for chunk in input_text.split("\n\n"):  # Split into manageable chunks
        summary = pipe_sum(chunk)
        results.append(summary[0]["summary_text"])
    return " ".join(results)

# Function to display PDF in the app
def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f"""
    <iframe
        src="data:application/pdf;base64,{base64_pdf}"
        width="100%"
        height="600"
        style="border: none;"
    ></iframe>
    """
    st.components.v1.html(pdf_display, height=600)

# Streamlit app
st.set_page_config(layout="wide")

def main():
    st.title("Document Summarization App using Language Model")
    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

    # Ensure directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if uploaded_file is not None:
        # Save the uploaded file
        file_path = os.path.join(data_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        st.info("Uploaded File")
        display_pdf(file_path)

        if st.button("Summarize"):
            with st.spinner("Summarizing..."):
                input_text = file_preprocessing(file_path)
                summary = llm_pipeline(input_text)
                st.success("Summarization Complete!")
                st.write(summary)

if __name__ == "__main__":
    main()
