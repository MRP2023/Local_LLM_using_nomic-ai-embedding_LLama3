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
import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes
import io
import pandas as pd

CHROMA_PATH = "chroma_db"
client = chromadb.HttpClient(host='localhost', port=8000)


def clear_vectorstore():
    collections = client.list_collections()

    if not collections:
        st.info("ℹ️ No collections found — nothing to clear.")
        return
    
    deleted_names = []
    for collection in collections:
        client.delete_collection(collection.name)
        deleted_names.append(collection.name)

    # if os.path.exists(CHROMA_PATH):
    #     shutil.rmtree(CHROMA_PATH)

    st.session_state.vectorstore = None
    st.session_state.chat_history = []
    st.success("🧹 Knowledge base cleared successfully!")

    # 🔄 Log directly instead of nesting expanders
    st.markdown("### 🧾 Deleted Collections Log:")
    for name in deleted_names:
        st.markdown(f"- `{name}`")

def detect_dynamic_header_row(table):
    def score_row(row):
        score = 0
        for cell in row:
            if cell:
                text = str(cell).strip()
                if text.replace(".", "").isdigit():
                    score -= 1
                else:
                    score += 1
        return score

    best_score = float('-inf')
    best_index = 0

    for i, row in enumerate(table):
        score = score_row(row)
        if score > best_score:
            best_score = score
            best_index = i

    return best_index



def extract_pdf_text_and_tables(pdf_path):
    documents = []
    logs = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # Extract plain text
            text = page.extract_text()
            if text:
                documents.append(Document(page_content=text, metadata={"page": page_num}))

            # Extract tables
            tables = page.extract_tables()
            for table_idx, table in enumerate(tables):
                if not table or len(table) < 2:
                    continue

                header_index = detect_dynamic_header_row(table)
                header = table[header_index]
                rows = table[header_index + 1:]

                # Store log entry
                logs.append(f"📄 Page {page_num} | 🧮 Table {table_idx} | 🏷️ Header row index: {header_index}")
                logs.append(f"🔑 Header columns: {header}")

                 # 🟩 Log detected header info
                print(f"[PAGE {page_num} | TABLE {table_idx}] Detected header row at index: {header_index}")
                print(f"[HEADER CONTENT]: {header}")

                try:
                    df = pd.DataFrame(rows, columns=header)
                except Exception as e:
                    print(f"[WARNING] Page {page_num}: {e}")
                    logs.append(f"⚠️ Page {page_num} Table {table_idx}: Failed to parse DataFrame: {e}")
                    continue

                # Convert each row to a readable document
                for idx, row in df.iterrows():
                    row_data = []
                    for col, val in row.items():
                        if pd.notnull(val) and pd.notnull(col):
                            row_data.append(f"{col.strip()}: {str(val).strip()}")
                    if row_data:
                        documents.append(Document(
                            page_content="\n".join(row_data),
                            metadata={"page": page_num, "type": "table_row"}
                        ))
    return documents, logs

def extract_ocr_text(pdf_path):
    """Fallback OCR method if text extraction fails."""
    with open(pdf_path, "rb") as f:
        images = convert_from_bytes(f.read())  # reads bytes from file
    ocr_documents = []
    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img)
        if text.strip():
            ocr_documents.append(Document(page_content=text, metadata={"page": i, "ocr": True}))
    return ocr_documents


def load_documents(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    return chunks


# def confirm_and_load_documents(pdf_path):
#     documents = extract_pdf_text_and_tables(pdf_path)
#     if not documents:
#         documents= extract_ocr_text(pdf_path)
#     if documents:
#         splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#         chunks = splitter.split_documents([doc for doc in documents if "TABLE_START" not in doc.page_content])
#         table_chunks = [doc for doc in documents if "TABLE_START" in doc.page_content]
#         return chunks+table_chunks
#     return []

def confirm_and_load_documents(pdf_path):
    documents, logs = extract_pdf_text_and_tables(pdf_path)
    if not documents:
        documents = extract_ocr_text(pdf_path)
        logs = ["ℹ️ OCR fallback used: no extractable text/tables found."]
    return documents, logs
        

def create_vectorstore(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embed_model = OllamaEmbeddings(model='nomic-embed-text:v1.5')
    db = Chroma.from_documents(
        chunks,
        embedding=embed_model,
        persist_directory=CHROMA_PATH, client=client
    )
    st.session_state.vectorstore = db
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

with st.expander("⚠️ Clear all data (vectorstore + chat)?", expanded=False):
    st.markdown("This will delete all previously stored PDF knowledge and reset the chat.")
    if st.button("🗑️ Yes, reset everything"):
        clear_vectorstore()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None




uploaded_file = st.file_uploader("📎 Upload your PDF", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    with st.spinner("🔍 Processing and indexing..."):
        # chunks = load_documents(temp_file_path)
        chunks, logs = confirm_and_load_documents(temp_file_path)
        vectorstore = create_vectorstore(chunks)
        st.success("✅ PDF processed and indexed!")

    # ✅ Show logs in the UI
    with st.expander("📋 Table Detection Logs"):
        for log in logs:
            st.markdown(log)

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
        
# else:
#     st.info("📥 Upload a PDF file to get started.", icon="📄")

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