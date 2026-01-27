# 🧠 NIMbleRAG

**NIMbleRAG** is a production-grade, high-performance Retrieval-Augmented Generation (RAG) agent designed to be "nimble" yet powerful. It leverages **[NVIDIA NIM](https://build.nvidia.com/)** (NVIDIA Inference Microservices) to deliver fast, secure, and accurate AI document interaction.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![NVIDIA NIM](https://img.shields.io/badge/AI-NVIDIA%20NIM-green)
![LangChain](https://img.shields.io/badge/Framework-LangChain-orange)

## 🚀 Why NIMbleRAG?

Standard RAG systems often suffer from poor retrieval accuracy or hallucinations. NIMbleRAG solves this by implementing an advanced "Retrieve & Refine" pipeline:

1.  **Hybrid Search**: It searches your documents using both **Keywords** (BM25) and **Semantics** (Vector) to find every relevant detail.
2.  **Reranking**: It uses a specialized model to double-check search results, discarding irrelevant noise before the LLM even sees them.
3.  **Accuracy**: By the time the context reaches the NVIDIA NIM LLM, it is highly relevant, leading to precise, cited answers.

## ✨ Key Techniques & Features

*   **🧠 NVIDIA NIM Integration**: Powered by state-of-the-art models like **Llama 3.1 8B** via [build.nvidia.com](https://build.nvidia.com/) for low-latency inference.
*   **🔍 Hybrid Search (The "Brain")**: Combines **BM25Retriever** (Sparse) and **FAISS** (Dense) to capture both exact terminology and conceptual meaning.
*   **🎯 Cross-Encoder Reranking**: Utilizes `ms-marco-MiniLM-L6-v2` to re-score the top-k retrieved documents. This acts as a quality filter, prioritizing the most relevant chunks.
*   **📂 Multi-Knowledgebase Architecture**: Create and manage isolated vector databases for different projects, reports, or domains (e.g., "Finance", "Legal", "Tech").
*   **📄 Intelligent Ingestion**: Uses **PyMuPDF** for robust PDF parsing and **RecursiveCharacterTextSplitter** for context-aware chunking.
*   **📑 Precise Citations**: Returns exact source filenames and **page numbers** for every generated answer, ensuring verifiability.

## 🛠️ Technology Stack

-   **LLM**: Meta Llama 3.1 8B (via [NVIDIA NIM](https://build.nvidia.com/))
-   **Embeddings**: [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (Running Locally)
-   **Reranker**: [`ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) (Running Locally)
-   **Vector DB**: FAISS (Facebook AI Similarity Search)
-   **Orchestration**: LangChain
-   **UI**: Streamlit (Multi-page App)
-   **Evaluation**: Ragas (Faithfulness, Answer Relevance metrics)

## 🚀 Setup & Installation

### Option 1: Local Development

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd NVIDIA-NIM-PDF-RAG
    ```

2.  **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment**:
    Create a `.env` file in the root directory and add your key from [build.nvidia.com](https://build.nvidia.com/):
    ```env
    NVIDIA_API_KEY=api_key
    ```

5.  **Run the Application**:
    ```bash
    streamlit run app.py
    ```

### Option 2: Docker Deployment

1.  **Build the Image**:
    ```bash
    docker build -t nimblerag .
    ```

2.  **Run the Container**:
    ```bash
    docker run -p 8501:8501 --env-file .env nimblerag
    ```
    Access the app at `http://localhost:8501`.

## 📂 Project Structure

```text
├── app.py                      # Main Landing Page & Navigation
├── pages/
│   ├── 1_Creating_Knowledgebase.py # Admin: Ingest & Index PDFs
│   └── 2_Chat_With_Data.py         # Chat: Select DB & Query
├── src/
│   ├── config.py              # Centralized Configuration
│   ├── document_processor.py  # PDF Parsing & Chunking
│   ├── retrieval_engine.py    # Hybrid Search & Reranking Implementation
│   ├── vector_manager.py      # Multi-DB Directory Management
│   ├── llm_chain.py           # LangChain Pipeline Builder
│   └── evaluation.py          # Ragas Evaluation Script
├── vector_dbs/                # Storage for Vector Indices
├── requirements.txt
└── Dockerfile
```
