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
```
### Step 2: Set Up the Environment
#### 1. Install Python (version 3.8+ recommended).
#### 2. Create a virtual environment:
```bash
python -m venv venv
```
#### 3. Activate the virtual environment:
<li>
  For macOS/Linux:
</li>

```bash
source venv/bin/activate
```
<li>
  For Windows:
</li>

```bash
venv\Scripts\activate
```
### Step 4: Install the required dependencies
```bash
pip install -r requirements.txt
```
## Step 3: Set Up Google Generative AI API Key
### 1.Create a `.env` file in the root of the project.
### 2.Add your Google API Key to the `.env` file:
```bash
GOOGLE_API_KEY=your-api-key-here
```

## Step 4: Run the Streamlit App
```bash
streamlit run app.py
```

## How to Use Files

Once the environment is set up and the app is running, you can upload and interact with PDF, Word, and Excel files for either Q&A or data analysis. Follow these steps:

#### Upload Files
1. **Navigate to the Sidebar**: On the left side of the app, you will see a "Menu" section where you can upload files.

2. **Upload Your Files**:
   - Click the "Upload Files (PDF, Excel, Word)" button.
   - Select one or multiple files to upload. The supported file formats are:
     - **PDF**
     - **Word Document (.docx)**
     - **Excel Spreadsheet (.xlsx, .xls)**

#### Select the Mode
After uploading your files, you can choose one of the following modes:
- **Q&A**: Interact with the content of the files by asking questions.
- **Data Analysis with Excel**: Perform basic data analysis on the uploaded Excel files.

#### Q&A Mode
1. **Ask a Question**: 
   - After selecting the Q&A mode, type your question in the text input box.
   - The app will use the content of your uploaded files to generate an answer based on the most relevant information.

2. **Submit the Question**: 
   - Click the "Submit & Process" button. The app will process the uploaded files and answer your question.

#### Data Analysis with Excel Mode
1. **View Excel Data**: 
   - The app will display the first few rows of each uploaded Excel sheet.
   - It will also show summary statistics such as the mean, median, and standard deviation of numerical columns.

2. **Comparison of DataFrames**: 
   - If you upload multiple Excel files, the app will compute the correlation between matching columns in the files and display the results.

---

That's it! You are now ready to interact with your files, whether for Q&A or data analysis.
