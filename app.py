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

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to extract text from Excel
def get_excel_text(excel_files):
    text = ""
    for file in excel_files:
        df = pd.read_excel(file)
        text += df.to_string(index=False)
    return text

# Function to extract text from Word
def get_word_text(word_docs):
    text = ""
    for doc in word_docs:
        document = Document(doc)
        for para in document.paragraphs:
            text += para.text + "\n"
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

# Function to set up the conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context, just say, "Answer is not available in the context." 
    Don't provide incorrect information.

    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user questions
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

# Main function
def main():
    st.set_page_config("Chat with Files")
    st.header("Chat with Your Files using Gemini 💁")

    user_question = st.text_input("Ask a Question from the Uploaded Files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True, type=["pdf"])
        excel_files = st.file_uploader("Upload Excel Files", accept_multiple_files=True, type=["xlsx", "xls"])
        word_docs = st.file_uploader("Upload Word Documents", accept_multiple_files=True, type=["docx"])

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = ""
                if pdf_docs:
                    raw_text += get_pdf_text(pdf_docs)
                if excel_files:
                    raw_text += get_excel_text(excel_files)
                if word_docs:
                    raw_text += get_word_text(word_docs)

                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
