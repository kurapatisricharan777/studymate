import os
import streamlit as st
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# --- Main App Configuration ---
st.set_page_config(page_title="StudyMate: AI Q&A Assistant", layout="wide")

# --- Function Definitions ---

def get_pdf_text(pdf_docs):
    """Extracts text from a list of uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        doc = fitz.open(stream=pdf.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()
    return text

def get_text_chunks(text):
    """Splits text into manageable chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks

@st.cache_resource
def get_vector_store(_text_chunks):
    """Creates and returns a FAISS vector store from text chunks."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(_text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain(api_key):
    """Creates and returns a LangChain QA chain with a custom prompt and Google's Gemini model."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the provided context, just say, "The answer is not available in the context".
    Do not provide a wrong answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = GoogleGenerativeAI(
        model="gemini-1.5-flash", 
        google_api_key=api_key,
        temperature=0.0
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    """Handles user input, performs similarity search, and gets the answer."""
    if "vector_store" not in st.session_state or st.session_state.vector_store is None:
        st.warning("Please upload and process your PDFs first.")
        return
    
    if "api_key" not in st.session_state or not st.session_state.api_key:
        st.warning("API Key not found. Please enter it in the sidebar.")
        return

    vector_store = st.session_state.vector_store
    docs = vector_store.similarity_search(user_question, k=3)
    
    chain = get_conversational_chain(st.session_state.api_key)
    
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    
    st.write("### Answer")
    st.write(f"{response['output_text']}")

# --- Streamlit UI ---

# Sidebar for API key and file uploads
with st.sidebar:
    st.title("📚 StudyMate Menu")
    st.write("Upload your academic PDFs and ask questions to get instant, context-aware answers.")
    
    st.header("1. Setup Your Credentials")
    
    # Use session_state to hold the API key
    api_key_input = st.text_input(
        "Enter your Google API Key:", 
        type="password",
        key="api_key_input" # A unique key for the widget
    )

    st.header("2. Upload Your PDFs")
    pdf_docs = st.file_uploader(
        "Upload your PDF files here and click on 'Process'",
        accept_multiple_files=True,
        type="pdf"
    )

    if st.button("Process Documents"):
        # On button click, update the session_state with the input value
        st.session_state.api_key = api_key_input
        
        if not st.session_state.api_key:
            st.warning("Please provide your Google API Key first.")
        elif pdf_docs:
            with st.spinner("Processing documents... this may take a moment."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                st.session_state.vector_store = get_vector_store(text_chunks)
                st.success("Processing complete! You can now ask questions.")
        else:
            st.warning("Please upload at least one PDF file.")

# Main content area
st.title("StudyMate: Your AI-Powered PDF Assistant 🤖")
st.markdown("---")

st.header("Ask a Question About Your Documents")
user_question = st.text_input("Enter your question here:")

if user_question:
    user_input(user_question)
