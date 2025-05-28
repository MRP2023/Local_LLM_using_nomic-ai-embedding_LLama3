# Local_LLM_using_nomic-ai-embedding_LLama3
🧠 A Streamlit-based AI assistant that lets you upload PDFs, extract tables and text (with OCR fallback), and chat with your documents using LLMs + vector search via ChromaDB and Nomic embeddings.

# 📄 Chat with Your PDF – AI-powered Q&A on Documents

This project allows you to upload PDF files (with structured text or scanned tables), extract their content using `pdfplumber`, `pandas`, and OCR (via `pytesseract`), and then ask questions directly about the document. Answers are powered by vector-based retrieval (using [ChromaDB](https://www.trychroma.com/)) and an LLM backend (e.g., LLaMA3 via Ollama).

---

## 🔧 Features

- 📎 Upload PDFs with plain text, scanned text, or tabular data
- 🧠 Automatic text and table extraction using `pdfplumber` + `pandas`
- 🖼️ OCR fallback for image-based PDFs (via `pytesseract`)
- 🤖 Ask natural language questions about the PDF
- 🧩 Nomic Embeddings + ChromaDB vector storage
- 🧼 Reset knowledge base with a single click
- 📋 Logs extracted table headers for debugging

---

## 📦 Stack

- [Streamlit](https://streamlit.io/) – UI framework
- [LangChain](https://www.langchain.com/) – RetrievalQA, chaining
- [ChromaDB](https://www.trychroma.com/) – Vector database
- [Ollama](https://ollama.com/) – Local LLMs like LLaMA 3
- [Nomic Embeddings](https://docs.nomic.ai/) – Open-source text embeddings
- `pandas`, `pdfplumber`, `pytesseract`, `pdf2image` – For PDF and table processing

---

## 🚀 Quick Start

```bash
# Clone the repo
git@github.com:MRP2023/Local_LLM_using_nomic-ai-embedding_LLama3.git

# Install dependencies
pip install -r requirements.txt

# (Make sure Docker-based ChromaDB is running)
# e.g.:
# docker run -v ./chroma-data:/data -p 8000:8000 chromadb/chroma

# Start the app
streamlit run app_ocr_table_using_panda.py
