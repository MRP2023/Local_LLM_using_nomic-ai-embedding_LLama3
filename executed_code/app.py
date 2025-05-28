import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
# from langchain.embeddings import NomicEmbedding
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import ollama
from langchain.chains import RetrievalQA
from langchain_community.llms.ollama import Ollama
import tempfile
import streamlit as st
import chromadb

CHROMA_PATH = "chroma_db"
client = chromadb.HttpClient(host='localhost', port=8000)

def load_documents(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    return chunks

def create_vectorstore(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embed_model = OllamaEmbeddings(model='nomic-embed-text:v1.5')
    db = Chroma.from_documents(
        chunks,
        embedding=embed_model,
        persist_directory=CHROMA_PATH, client=client
    )
    db.persist()
    return db

def load_vectorstore():
    # embed_model = NomicEmbedding(model="nomic-embed-text-v1")
    embed_model = OllamaEmbeddings(model='nomic-embed-text:v1.5')
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embed_model, client=client)

def ask_question(vectorstore, question):
    retriever = vectorstore.as_retriever()
    embed_model = OllamaEmbeddings(model='nomic-embed-text:v1.5')
    # llm = ollama(model="llama3:latest")
    llm= Ollama(model="llama3")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = qa_chain.run(question)
    return result



# --- Streamlit UI ---
st.set_page_config(page_title="📄 PDF Q&A Chatbot", layout="centered")
st.title("🧠 Chat with Your Documents with Python Agent 🐍 ")
st.markdown("Upload any kind of PDF  document and chat with it your user friendly Python Snake ")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


uploaded_file = st.file_uploader("📎 Upload your PDF", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    with st.spinner("🔍 Processing and indexing..."):
        chunks = load_documents(temp_file_path)
        vectorstore = create_vectorstore(chunks)
        st.success("✅ PDF processed and indexed!")

    os.remove(temp_file_path)

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.text_input("💬 Ask something from your Document")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("🐍"):
            with st.spinner("🤖 Python is thinking..."):
                vectorstore = load_vectorstore()
                result = ask_question(vectorstore, question)
                st.markdown(f"**{result}**")
                st.session_state.chat_history.append({"role": "assistant", "content": f"**{result}**"})
                # st.success("💡 Answer:")
                # st.write(result)
            
else:
    st.info("📥 Upload a PDF file to get started.", icon="📄")

st.markdown(
    """
    <style>
        .credit {
            position: fixed;
            bottom: 10px;
            right: 15px;
            font-size: 14px;
            color: green;
            z-index: 100;
        }
        .credit a {
            text-decoration: none;
            color: green;
        }
        .credit a:hover {
             color: darkgreen;
        }
    </style>
    <div class="credit">
        Developed by <a href="https://github.com/MRP2023" target="_blank">Khandoker Md. Mashiur Rahman Pranto</a>
    </div>
    """,
    unsafe_allow_html=True
)