# 🧠 NVIDIA NIM PDF RAG

A powerful **Retrieval-Augmented Generation (RAG)** application designed to let you chat with your PDF documents. Built with **LangChain**, **NVIDIA NIM**, and **Streamlit**, this project demonstrates a modern, efficient way to extract insights from documents with source citations.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=LangChain&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

---

## ✨ Features

- **Advanced PDF Parsing**: Uses **PyMuPDF (fitz)** for high-fidelity text extraction, preserving layout semantics better than standard parsers.
- **Source Citations**: Every answer includes a "View Sources" expandable section, showing exactly which matching documents (and **Page Numbers**) were used to generate the response.
- **Hybrid Search**: Leverages `all-MiniLM-L6-v2` local embeddings with **FAISS** for fast, local vector search.
- **State-of-the-Art LLM**: Powered by **NVIDIA NIM (Llama 3.1 8B Instruct)** for high-quality, context-aware answers.
- **Modern UI**:
    - **Streaming Responses**: Watch the answer type out in real-time.
    - **Chat Interface**: Fully distinct user/assistant message bubbles.
    - **Session Management**: Clear conversation history and document context with a single click.

---

## 🏗️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **LLM Provider**: [NVIDIA NIM](https://build.nvidia.com/explore/discover) (via `langchain-nvidia-ai-endpoints`)
- **Embeddings**: `HuggingFaceEmbeddings` (Local inference)
- **Vector DB**: [FAISS](https://github.com/facebookresearch/faiss) (Local file-based)
- **PDF Parser**: [PyMuPDF](https://pymupdf.readthedocs.io/)
- **Framework**: [LangChain](https://www.langchain.com/)

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+ installed
- An NVIDIA API Key (Get it [here](https://build.nvidia.com/explore/discover))

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone <your-repo-url>
cd NVIDIA-NIM-PDF-RAG

# Create a virtual environment (Recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the root directory and add your NVIDIA API Key:

```env
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. Running the App

Run the Streamlit application:

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 🐳 Docker Support

You can also run this application in a container.

**Build the image:**
```bash
docker build -t rag-pdf-app .
```

**Run the container:**
```bash
docker run --rm -p 8501:8501 --env-file .env rag-pdf-app
```

---

## 📂 Project Structure

```bash
├── app.py                 # Main Streamlit application
├── requirements.txt       # Project dependencies
├── Dockerfile             # Docker configuration
├── .env                   # Environment variables (API Keys)
├── src/
│   └── utils.py           # Core RAG logic (PDF parsing, Embeddings, Chain)
├── faiss_db/              # Local Vector DB storage (Auto-generated)
└── README.md              # Project Documentation
```

## 📝 How it Works

1.  **Ingest**: `PyMuPDF` reads the uploaded PDF and splits it into pages.
2.  **Chunk**: `RecursiveCharacterTextSplitter` breaks pages into manageable text chunks (1000 chars) while keeping metadata (Page #).
3.  **Embed**: `HuggingFaceEmbeddings` converts chunks into vector representations.
4.  **Store**: `FAISS` indexes these vectors for fast similarity search.
5.  **Retrieve**: When you ask a question, the system finds the top K most relevant chunks.
6.  **Generate**: The retrieved text + your question are sent to the **NVIDIA NIM LLM**, which generates a referenced answer.

---

## 🤝 Contributing
Feel free to fork this project and submit Pull Requests. Suggestions for new features (e.g., Local LLM support via Ollama) are welcome!
