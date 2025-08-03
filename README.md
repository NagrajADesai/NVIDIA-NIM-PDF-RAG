# NVIDIA-NIM-PDF-RAG

## 🧠 RAG-Powered PDF Q&A Application

This is a Retrieval-Augmented Generation (RAG) application that allows you to ask questions over the content of uploaded PDF files. It uses:

- **LangChain** for chaining components
- **FAISS** for vector-based document retrieval
- **MiniLM Embeddings** via HuggingFace
- **NVIDIA LLM API** (e.g., LLaMa 3.1) for generating responses
- **Streamlit** as the front-end interface

---

## ⚙️ Prerequisites

- Docker installed: [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)
- `.env` file with your API key:

NVIDIA_API_KEY=your_nvidia_api_key_here

Place this file in the same directory as your `app.py`.

---

## 🐳 Build Docker Image

Open a terminal in your project root and run:

```bash
docker build -t rag-pdf-app .
```

## Run Docker Container

To start the app in a container and expose it on port 8501:

```
docker run --rm -p 8501:8501 --env-file .env rag-pdf-app
```

## Usage

Once running, open your browser and visit:
http://localhost:8501
