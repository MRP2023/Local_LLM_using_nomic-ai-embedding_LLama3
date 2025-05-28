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

CHROMA_PATH = "chroma_db"
client = chromadb.HttpClient(host='localhost', port=8000)

######## Working for Normal Table ########

# def extract_pdf_text_and_tables(pdf_file):
#     """Extract both raw text and structured table data."""
#     documents = []

#     with pdfplumber.open(pdf_file) as pdf:
#         for i, page in enumerate(pdf.pages):
#             text = page.extract_text()
#             if text:
#                 documents.append(Document(page_content=text, metadata={"page": i}))
            
#             table = page.extract_table()
#             if table:
#                 # formatted_rows = [" | ".join(row) for row in table if row]

#                 formatted_rows = [" | ".join(str(cell) for cell in row) for row in table if row]
#                 table_text = "\n".join(formatted_rows)
#                 # documents.append(Document(page_content=formatted_table, metadata={"page": i, "table": True}))
#                 print(f"[DEBUG][Page {i}] Extracted Table:\n{table_text}")
#                 documents.append(Document(page_content=f"TABLE_START\n{table_text}\nTABLE_END", metadata={"page": i, "table": True}))
#                 # print(documents)
    
#     return documents

######## Working for Normal Table (End) ########



# def extract_pdf_text_and_tables(pdf_file):
#     """Extract both raw text and structured table data."""
#     documents = []

#     with pdfplumber.open(pdf_file) as pdf:
#         for i, page in enumerate(pdf.pages):
#             text = page.extract_text()
#             if text:
#                 documents.append(Document(page_content=text, metadata={"page": i}))
            
#             tables = page.extract_tables()
#             if tables:

#                 for t_idx,table in enumerate(tables):
#                     if not table or len(table) < 2:
#                         continue

#                     header = table[0]
#                     if not isinstance(header, list) or any(h is None or len(h) <= 1 for h in header):
#                         print(f"[WARNING] Skipping malformed header on Page {i}, Table {t_idx}: {header}")
#                         continue

#                     for row in table[1:]:
#                         if not row or not any(row):
#                             continue
#                         try:
#                             row_dict = dict(zip(header, row))
#                             lines = [f"{k.strip()}: {v.strip() if v else 'N/A'}" for k, v in row_dict.items()]
#                             formatted = "\n".join(lines)
#                             print(f"[DEBUG][Page {i}] Extracted Table:\n{formatted}")
#                             documents.append(Document(page_content=formatted, metadata={"page": i, "type": "table"}))
#                         except Exception as e:
#                             print(f"[ERROR] Failed to process row on page {i}: {e}")
#                             continue
    
#     return documents

def extract_pdf_text_and_tables(pdf_file):
    """Extract both raw text and structured table data."""
    documents = []

    with pdfplumber.open(pdf_file) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                documents.append(Document(page_content=text, metadata={"page": i}))
            
            table = page.extract_table()
            # print(table)
            if table:
                # formatted_rows = [" | ".join(row) for row in table if row]

                formatted_rows = [" | ".join(str(cell) for cell in row) for row in table if row]
                table_text = "\n".join(formatted_rows)
                # documents.append(Document(page_content=formatted_table, metadata={"page": i, "table": True}))
                # print(f"[DEBUG][Page {i}] Extracted Table:\n{table_text}")
                documents.append(Document(page_content=f"TABLE_START\n{table_text}\nTABLE_END", metadata={"page": i, "table": True}))
                print(documents)
    
    return documents

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


def confirm_and_load_documents(pdf_path):
    documents = extract_pdf_text_and_tables(pdf_path)
    if not documents:
        documents= extract_ocr_text(pdf_path)
    if documents:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents([doc for doc in documents if "TABLE_START" not in doc.page_content])
        table_chunks = [doc for doc in documents if "TABLE_START" in doc.page_content]
        return chunks+table_chunks
    return []
        

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
        # chunks = load_documents(temp_file_path)
        chunks = confirm_and_load_documents(temp_file_path)
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