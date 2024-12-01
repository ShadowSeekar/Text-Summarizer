import streamlit as st 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
from PyPDF2 import PdfReader
import torch
import base64


checkpoint = "MBZUAI/LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)

def file_preprocessing(file):
    loader =  PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        print(text)
        final_texts = final_texts + text.page_content
    return final_texts

def llm_pipeline(filepath):
    pipe_sum = pipeline(
        'summarization',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 500, 
        min_length = 50)
    input_text = file_preprocessing(filepath)
    result = pipe_sum(input_text)
    result = result[0]['summary_text']
    return result

@st.cache_data

def displayPDF(file):
    #if file is not None:
    pdf_reader = PdfReader(file) # read your PDF file
    # extract the text data from your PDF file after looping through its pages with the .extract_text() method
    text_data= ""
    for page in pdf_reader.pages: # for loop method
        text_data+= page.extract_text()
    st.write(txt_data)
#    with open(file, "rb") as f:
#        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
#    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
#    st.markdown(pdf_display, unsafe_allow_html=True)

st.set_page_config(layout="wide")

import os

def main():
    st.title("Document Summarization App using Language Model")
    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

    # Ensure the directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if uploaded_file is not None:
        if st.button("Summarize"):
            col1, col2 = st.columns(2)

            # Save the uploaded file to the directory
            filepath = os.path.join(data_dir, uploaded_file.name)
            with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())

            with col1:
                st.info("Uploaded File")
                pdf_view = displayPDF(uploaded_file)
                

            with col2:
                summary = llm_pipeline(filepath)
                st.info("Summarization Complete")
                st.success(summary)







if __name__ == "__main__":
    main()
