# Chat with PDF, Word, and Excel Files 📄

This project allows users to interact with their uploaded files (PDF, Word, and Excel) to either perform Q&A or analyze data, leveraging the power of AI and machine learning. The application uses a combination of natural language processing (NLP) and document processing techniques to provide detailed responses to user questions or analyze Excel data for insights.

## Features:
- **File Upload**: Upload PDF, Word, and Excel files.
- **Text Extraction**:
  - Extracts text from PDFs using `PyPDF2`.
  - Extracts text from Word documents using `python-docx`.
  - Extracts text from Excel files using `pandas`.
- **Text Chunking**: Splits large text into manageable chunks using `langchain`’s `RecursiveCharacterTextSplitter`.
- **Vector Store**: Stores text embeddings for similarity search using `FAISS` and `GoogleGenerativeAIEmbeddings`.
- **Q&A Functionality**: Ask questions based on the context of the uploaded files.
- **Excel Data Analysis**: View basic statistics and correlations between Excel sheets.

## Libraries & Technologies:
1. **Streamlit**: 
   - Used for building the web interface of the project. It provides an easy-to-use framework to display the uploaded files, handle user input, and show responses interactively.
   - [Streamlit Documentation](https://docs.streamlit.io/)

2. **PyPDF2**: 
   - This library is used for extracting text from PDF files.
   - [PyPDF2 Documentation](https://pythonhosted.org/PyPDF2/)

3. **python-docx**: 
   - Used for reading and extracting text from Word documents.
   - [python-docx Documentation](https://python-docx.readthedocs.io/en/latest/)

4. **Pandas**:
   - Essential for handling and analyzing Excel files. It is used to read Excel data and perform basic analysis like generating summary statistics and finding correlations between sheets.
   - [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)

5. **langchain**:
   - Provides tools for text splitting, embedding generation, and chaining models together for question answering.
   - [langchain Documentation](https://langchain.readthedocs.io/en/latest/)

6. **FAISS**:
   - A library for efficient similarity search and clustering of dense vectors. It is used to store and retrieve text embeddings from the uploaded files.
   - [FAISS Documentation](https://github.com/facebookresearch/faiss)

7. **Google Generative AI**:
   - Used for generating text embeddings and for answering questions by analyzing the content of uploaded documents.
   - [Google Generative AI Documentation](https://developers.google.com/ai/generative)

8. **dotenv**:
   - Manages environment variables such as API keys securely.
   - [dotenv Documentation](https://pypi.org/project/python-dotenv/)

## Flow of the Project:

1. **File Upload**:
   - Users upload one or more files (PDF, Word, or Excel) via the web interface.
   
2. **Text Extraction**:
   - Depending on the file format, the corresponding text extraction method is applied:
     - **PDF**: Text is extracted using `PyPDF2`.
     - **Word**: Text is extracted using `python-docx`.
     - **Excel**: Text is extracted using `pandas`.

3. **Text Chunking**:
   - The extracted text is split into manageable chunks using `langchain`'s `RecursiveCharacterTextSplitter`. This step is crucial for large documents to prevent memory issues and optimize processing.

4. **Vector Store Creation**:
   - The text chunks are embedded using `GoogleGenerativeAIEmbeddings`. The embeddings are stored in a `FAISS` vector store for similarity search, enabling efficient retrieval of relevant information based on user queries.

5. **Q&A Functionality**:
   - The user can ask questions about the uploaded files. The system performs a similarity search to retrieve relevant document chunks and provides answers using `GoogleGenerativeAI`'s conversational model.

6. **Excel Data Analysis**:
   - If the uploaded file is an Excel file, the system provides basic data analysis (such as summary statistics) and compares columns between sheets.

## How to Clone and Run the Project:

Follow the steps below to clone and run the project on your local system:

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/chat-with-files.git
cd chat-with-files
