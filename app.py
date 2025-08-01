from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from operator import itemgetter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableMap

import streamlit as st
import tempfile
import os
import datetime
import uuid
import time
import gspread
from google.oauth2.service_account import Credentials

# Get API keys
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

# Set environment variables
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY if GEMINI_API_KEY else ""
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY if OPENAI_API_KEY else ""

st.set_page_config(page_title="DocPrise: File QA Chatbot", page_icon="🔍")
st.title("🎯 Surprise Yourself! Let us handle where your information is")

# Generate unique session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

# Google Sheets logging function
# Updated Google Sheets logging function with proper scopes
def log_to_google_sheets(question, answer, response_time=None):
    """Log user interactions to Google Sheets with proper authentication"""
    try:
        # Get credentials from Streamlit secrets
        creds_dict = st.secrets["gcp_service_account"]
        
        # Add the correct scopes for Google Sheets and Drive
        credentials = Credentials.from_service_account_info(
            creds_dict,
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive.file",
                "https://www.googleapis.com/auth/drive"
            ]
        )
        
        # Connect to Google Sheets
        gc = gspread.authorize(credentials)
        
        # Open the spreadsheet (or create if doesn't exist)
        try:
            sheet = gc.open("DocPrise Analytics").sheet1
        except gspread.SpreadsheetNotFound:
            # Create new spreadsheet and share it
            spreadsheet = gc.create("DocPrise Analytics")
            sheet = spreadsheet.sheet1
            
            # Add headers
            sheet.append_row([
                "Timestamp", "Session_ID", "Question", "Answer", 
                "Question_Length", "Answer_Length", "Response_Time"
            ])
            
            # Make it accessible (optional - makes it viewable by anyone with link)
            spreadsheet.share('', perm_type='anyone', role='reader')
        
        # Prepare data
        row_data = [
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            st.session_state.session_id,
            question,
            answer[:500] + "..." if len(answer) > 500 else answer,
            len(question),
            len(answer),
            round(response_time, 2) if response_time else None
        ]
        
        # Add row to sheet
        sheet.append_row(row_data)
        
        # Show success in sidebar
        st.sidebar.success("✅ Interaction logged!")
        
    except Exception as e:
        # Show detailed error for debugging
        st.sidebar.error(f"❌ Logging failed: {str(e)}")
        st.sidebar.error(f"Error type: {type(e).__name__}")

@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyMuPDFLoader(temp_filepath)
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    doc_chunks = text_splitter.split_documents(docs)

    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    vectordb = FAISS.from_documents(doc_chunks, embeddings_model)
    return vectordb.as_retriever()

# File upload
uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files", 
    type=["pdf"],
    accept_multiple_files=True
)


if not uploaded_files:
    st.info("Please upload PDF documents to continue.")
    st.stop()

# Configure retriever
with st.spinner("Processing documents..."):
    retriever = configure_retriever(uploaded_files)

# QA Template
qa_template = """
Use only the following pieces of context to answer the question at the end. 
If you don't know the answer, just say you don't know, don't try to make up the answer. 
Make your answers interactive, engaging and well-structured to make it easy to read, rather than long text, using bulletpoints, good formatting, some emojis when needed, but not much, not very often.

Context:
{context}

Question: {question}

Answer:
"""

qa_prompt = ChatPromptTemplate.from_template(qa_template)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# Initialize Gemini
gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.1,
    streaming=False,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Create RAG chain
qa_rag_chain = (
    RunnableMap({
        "context": itemgetter("question") | retriever | format_docs,
        "question": itemgetter("question")
    })
    | qa_prompt
    | gemini
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input with Google Sheets logging
if prompt := st.chat_input("Ask a question about your documents"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start_time = time.time()
            
            try:
                # Get response
                response = qa_rag_chain.invoke({"question": prompt})
                
                # Extract answer
                if hasattr(response, 'content'):
                    answer = response.content
                else:
                    answer = str(response)
                
                response_time = time.time() - start_time
                
                # Display the response
                st.markdown(answer)
                
                # Log to Google Sheets
                log_to_google_sheets(prompt, answer, response_time)
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer
                })
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                
                # Log error to Google Sheets too
                log_to_google_sheets(prompt, f"ERROR: {str(e)}", None)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg
                })

