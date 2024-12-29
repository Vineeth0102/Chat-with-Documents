import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pandas as pd
from docx import Document
import re  # For handling filename patterns if needed
import time  # Optional, for adding delays or time-based operations


load_dotenv()
os.getenv("GOOGLE_API_KEY")

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDFs
def extract_pdf_text(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from Word documents
def extract_word_text(file):
    text = ""
    doc = Document(file)
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Function to extract text from Excel sheets (concatenating all cells)
def extract_excel_text(file):
    text = ""
    df = pd.read_excel(file)
    for column in df.columns:
        text += " ".join(df[column].astype(str).tolist()) + "\n"
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save a vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to set up a conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. If the answer is not in
    the provided context, just say, "answer is not available in the context". Don't provide the wrong answer.

    Context:
    {context}?
    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and generate response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply:", response["output_text"])

# Function to analyze Excel data
def analyze_excel(files):
    dataframes = []
    for file in files:
        df = pd.read_excel(file)
        dataframes.append(df)
    
    # Example Analysis: Basic stats
    st.write("### Uploaded Excel Data:")
    for i, df in enumerate(dataframes):
        st.write(f"**Sheet {i + 1}:**")
        st.write(df.head())  # Display first few rows
        st.write("**Summary Statistics:")
        st.write(df.describe())  # Basic stats

    # Example: Compare DataFrames if more than one
    if len(dataframes) > 1:
        st.write("### Comparison:")
        for col in dataframes[0].columns:
            if col in dataframes[1].columns:
                st.write(f"**Correlation for column `{col}`:**")
                correlation = dataframes[0][col].corr(dataframes[1][col])
                st.write(f"Correlation: {correlation:.2f}")

# Universal function to extract or analyze files
def process_files(files, mode):
    if mode == "Data Analysis with Excel":
        st.write("Performing Excel data analysis...")
        analyze_excel(files)
        return None  # Skip text chunking and vector store creation
    else:
        text = ""
        for file in files:
            file_name = file.name
            if file_name.endswith(".pdf"):
                text += extract_pdf_text(file)
            elif file_name.endswith(".docx"):
                text += extract_word_text(file)
            elif file_name.endswith((".xlsx", ".xls")):
                text += extract_excel_text(file)
            else:
                st.warning(f"Unsupported file format: {file_name}")
        return text

def main():
    st.set_page_config("Chat with Files & Analyze Excel")
    st.header("Chat with Your Files or Analyze Excel 📊")

    # Dropdown menu for mode selection
    mode = st.selectbox("Select Mode:", ["Q&A", "Data Analysis with Excel"])

    user_question = ""
    processed_text = None  # Initialize with a default value

    if mode == "Q&A":
        user_question = st.text_input("Ask a Question from the Uploaded Files (if applicable)")

    with st.sidebar:
        st.title("Menu:")
        files = st.file_uploader(
            "Upload Files (PDF, Excel, Word)", 
            accept_multiple_files=True, 
            type=["pdf", "xlsx", "xls", "docx"]
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                processed_text = process_files(files, mode)
                if mode == "Q&A" and processed_text:
                    text_chunks = get_text_chunks(processed_text)
                    get_vector_store(text_chunks)
                    st.success("Text-based files processed successfully!")
                elif mode == "Data Analysis with Excel":
                    st.success("Excel data analysis completed!")

    # Handle Q&A separately
    if mode == "Q&A" and user_question:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question)
            chain = get_conversational_chain()
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            st.write("Reply:", response["output_text"])
        except Exception as e:
            st.error(f"Error processing your query: {str(e)}")
            st.warning("Make sure to upload and process the relevant files first.")


if __name__ == "__main__":
    main()

