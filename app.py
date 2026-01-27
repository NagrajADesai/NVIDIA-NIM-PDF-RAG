import streamlit as st
from src.config import AppConfig

def main():
    st.set_page_config(
        page_title=AppConfig.APP_TITLE,
        page_icon=AppConfig.APP_ICON,
        layout=AppConfig.LAYOUT
    )

    st.title("🧠 NIMbleRAG: Advanced RAG Agent")
    
    st.markdown("""
    ### 🚀 Production-Grade RAG System
    
    This application transforms static PDF documents into an intelligent, queryable knowledge base. It leverages cutting-edge **Retrieval-Augmented Generation (RAG)** techniques to ensure high-accuracy answers.

    #### ⚙️ Technical Architecture:
    
    -   **🧠 Generation**: Powered by **NVIDIA NIM** (Llama 3.1 8B).
    -   **🔍 Hybrid Search**: Combines **BM25** (Keyword Match) and **FAISS** (Semantic Match) for superior retrieval recall.
    -   **🎯 Reranking**: Uses **Cross-Encoders** (`ms-marco-MiniLM-L6-v2`) to refine and score the top retrieved results.
    -   **📂 Multi-Knowledgebase**: Create, manage, and query specific separate vector databases.
    -   **🔬 Evaluation**: Built-in support for **Ragas** metrics.

    #### 📚 How to Use:
    
    1.  **📂 Create Knowledgebase**:
        -   Navigate to the **"Creating Knowledgebase"** page.
        -   Upload your research papers, manuals, or contracts.
        -   The system will ingest, chunk, and index them using Hybrid Search.
    
    2.  **💬 Chat With Data**:
        -   Switch to the **"Chat With Data"** page.
        -   Select your target Knowledgebase.
        -   Experience the power of Reranked RAG with precise citations (including page numbers).
    
    ---
    *Built with LangChain, FAISS, Streamlit, and NVIDIA AI Endpoints.*
    """)

if __name__ == "__main__":
    main()