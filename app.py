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

# Get API keys
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

# Set environment variables
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY if GEMINI_API_KEY else ""
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY if OPENAI_API_KEY else ""

st.set_page_config(page_title="FileBit: File QA Chatbot", page_icon="🔍")
st.title("Think Different! Let us handle where your information is")

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
Keep the answer concise and well-structured to make it easy to read.

Context:
{context}

Question: {question}

Answer:
"""

qa_prompt = ChatPromptTemplate.from_template(qa_template)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# Initialize Gemini WITHOUT streaming
gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.1,
    streaming=False,  # Turn off streaming
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

# Chat input
if prompt := st.chat_input("Ask a question about your documents"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Get response (non-streaming)
                response = qa_rag_chain.invoke({"question": prompt})
                
                # Extract the actual text content
                if hasattr(response, 'content'):
                    answer = response.content
                else:
                    answer = str(response)
                
                # Display the response
                st.markdown(answer)
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer
                })
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg
                })
