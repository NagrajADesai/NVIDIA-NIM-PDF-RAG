import os
import pickle
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.docstore.document import Document
from sentence_transformers import CrossEncoder
from src.config import AppConfig, ModelConfig

class RetrievalEngine:
    """Handles Hybrid Search and Reranking."""

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=ModelConfig.EMBEDDING_MODEL)
        self.vector_store: Optional[FAISS] = None
        self.reranker = CrossEncoder(ModelConfig.RERANKER_MODEL)
        self.bm25_retriever: Optional[BM25Retriever] = None

    def initialize_vector_store(self, text_chunks: List[Document], save_path: str):
        """
        Initializes or upgrades variables for the vector store.
        """
        # Create FAISS Vector Store
        if os.path.exists(os.path.join(save_path, "index.faiss")):
             self.vector_store = FAISS.load_local(save_path, self.embeddings, allow_dangerous_deserialization=True)
             if text_chunks:
                 new_store = FAISS.from_documents(text_chunks, embedding=self.embeddings)
                 self.vector_store.merge_from(new_store)
                 self.vector_store.save_local(save_path)
        else:
             if text_chunks:
                self.vector_store = FAISS.from_documents(text_chunks, embedding=self.embeddings)
                self.vector_store.save_local(save_path)
        
        # Handle BM25 Retriever Persistence
        bm25_path = os.path.join(save_path, "bm25.pkl")
        
        if text_chunks:
             # Create and Save BM25
             self.bm25_retriever = BM25Retriever.from_documents(text_chunks)
             self.bm25_retriever.k = 10
             with open(bm25_path, "wb") as f:
                 pickle.dump(self.bm25_retriever, f)
        elif os.path.exists(bm25_path):
             # Load BM25
             with open(bm25_path, "rb") as f:
                 self.bm25_retriever = pickle.load(f)
             self.bm25_retriever.k = 10

    def get_hybrid_retriever(self):
        """Returns an EnsembleRetriever (BM25 + FAISS)."""
        if not self.vector_store or not self.bm25_retriever:
            raise ValueError("Retrievers not initialized. Please process documents first.")
        
        faiss_retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})
        
        ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )
        return ensemble_retriever

    def rerank_documents(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """
        Reranks retrieved documents using a Cross-Encoder.
        """
        if not documents:
            return []

        # Prepare pairs for cross-encoder
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.reranker.predict(pairs)
        
        # Attach scores to documents for debugging/sorting
        for doc, score in zip(documents, scores):
            doc.metadata["score"] = score

        # Sort by score descending
        sorted_docs = sorted(documents, key=lambda x: x.metadata["score"], reverse=True)
        
        return sorted_docs[:top_k]
