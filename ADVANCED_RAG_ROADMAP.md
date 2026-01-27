# 🚀 Advanced RAG Improvement Roadmap

This document outlines the steps to upgrade existing specific "NVIDIA-NIM-PDF-RAG" project into a production-grade **Advanced RAG** system.

## 1. Upgrade Retrieval Pipeline (The "Brain")

The current implementation uses simple dense vector search. To improve accuracy, especially for domain-specific queries, we need **Hybrid Search** and **Reranking**.

### A. Hybrid Search (Dense + Sparse)
**Why?** Dense vectors (embeddings) are great for meaning, but sometimes miss exact keyword matches (e.g., part numbers, specific acronyms).
**How?**
- **Current**: `FAISS` (Dense only).
- **Upgrade**: Combine `FAISS` (Dense) with `BM25` (Keyword/Sparse).
- **Implementation**: Use `EnsembleRetriever` from LangChain to weigh results from both retrievers (e.g., 0.5 Dense + 0.5 Sparse).

### B. Reranking (Refinement)
**Why?** The top-k results from vector search aren't always the best. A "Cross-Encoder" model looks at the Question-Passage pair closely to score relevance.
**How?**
- **Step**: Retrieve Top-20 docs (instead of Top-5).
- **Action**: Pass them through a Reranker using the local model `models--cross-encoder--ms-marco-MiniLM-L6-v2`.
- **Result**: Return the Top-5 *reranked* documents to the LLM. This drastically reduces hallucinations.

## 2. Ingestion Strategy (The "Input")

We will stick to the robust and simple **RecursiveCharacterTextSplitter**.
- **Strategy**: Keep using `RecursiveCharacterTextSplitter` with current parameters (or optimized chunk sizes).
- **Decision**: Do not implement Semantic Chunking or Parent Document Retriever at this stage.

## 3. Evaluation & Monitoring (The "Quality Control")

You cannot improve what you cannot measure.

### A. RAGAS (RAG Assessment)
**Why?** To know if your RAG is hallucinating or missing context.
**Upgrade**: Implement a script that uses `Ragas` to calculate:
- **Faithfulness**: Is the answer derived from the context?
- **Answer Relevance**: Does the answer address the question?
- **Context Recall**: Did we retrieve the right info?

## 4. UI/UX Enhancements

- **Source Citation**: Display the source document name and the specific PDF page number for the retrieved chunks.

---

## 🏗️ Implementation Priority (Suggested)

1. **Quick Wins**: Implement **Hybrid Search (BM25)** and **Cross-Encoder Reranking** (using `models--cross-encoder--ms-marco-MiniLM-L6-v2`).
2. **Evaluation**: Set up **RAGAS**.
3. **UI Updates**: Update source citations (include page numbers), remove chat history.
