from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks.base import BaseCallbackHandler
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

# Debug: Show API key status
st.sidebar.write("🔑 API Key Status:")
st.sidebar.write(f"Gemini: {'✅' if GEMINI_API_KEY else '❌'}")
st.sidebar.write(f"OpenAI: {'✅' if OPENAI_API_KEY else '❌'}")

# Set environment variables
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY if GEMINI_API_KEY else ""
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY if OPENAI_API_KEY else ""

st.set_page_config(page_title="FileBit: File QA Chatbot", page_icon="🔍")
st.title("Think Different! Let us handle where your information is")

@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    try:
        docs = []
        temp_dir = tempfile.TemporaryDirectory()
        
        for file in uploaded_files:
            temp_filepath = os.path.join(temp_dir.name, file.name)
            with open(temp_filepath, "wb") as f:
                f.write(file.getvalue())
            loader = PyMuPDFLoader(temp_filepath)
            docs.extend(loader.load())

        st.sidebar.write(f"📄 Loaded {len(docs)} document pages")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        doc_chunks = text_splitter.split_documents(docs)
        
        st.sidebar.write(f"📝 Created {len(doc_chunks)} text chunks")

        embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        vectordb = FAISS.from_documents(doc_chunks, embeddings_model)
        st.sidebar.write("✅ Vector database created")
        
        return vectordb.as_retriever()
    except Exception as e:
        st.error(f"Error in configure_retriever: {str(e)}")
        return None

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# File upload
uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files", 
    type=["pdf"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Please upload PDF documents to continue.")
    st.stop()

if not GEMINI_API_KEY or not OPENAI_API_KEY:
    st.error("❌ Missing API keys! Please add both GEMINI_API_KEY and OPENAI_API_KEY to your secrets.")
    st.stop()

# Configure retriever
with st.spinner("Processing documents..."):
    retriever = configure_retriever(uploaded_files)

if not retriever:
    st.error("Failed to process documents. Please check your files and try again.")
    st.stop()

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
    formatted = "\n\n".join([d.page_content for d in docs])
    st.sidebar.write(f"📋 Retrieved {len(docs)} relevant chunks")
    return formatted

# Initialize Gemini
try:
    gemini = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.1,
        streaming=True,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    st.sidebar.write("🤖 Gemini initialized")
except Exception as e:
    st.error(f"Failed to initialize Gemini: {str(e)}")
    st.stop()

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
        response_container = st.empty()
        stream_handler = StreamHandler(response_container)
        
        try:
            st.sidebar.write("🔄 Generating response...")
            
            # Test retrieval first
            docs = retriever.get_relevant_documents(prompt)
            st.sidebar.write(f"🔍 Found {len(docs)} relevant documents")
            
            # Get full response
            response = qa_rag_chain.invoke(
                {"question": prompt}, 
                config={"callbacks": [stream_handler]}
            )
            
            # Add assistant response to chat history
            final_response = stream_handler.text if stream_handler.text else "No response generated"
            st.session_state.messages.append({
                "role": "assistant", 
                "content": final_response
            })
            
            st.sidebar.write("✅ Response generated")
                
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            st.error(error_msg)
            st.sidebar.write(f"❌ {error_msg}")
            
            # Add error to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"Sorry, I encountered an error: {str(e)}"
            })
