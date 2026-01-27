import streamlit as st
import time
from src.config import AppConfig, ModelConfig
from src.document_processor import DocumentProcessor
from src.retrieval_engine import RetrievalEngine
from src.llm_chain import LLMChainBuilder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

def stream_text(text):
    """Yields text one character at a time for streaming effect."""
    for char in text:
        yield char
        time.sleep(0.005)

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed_docs" not in st.session_state:
        st.session_state.processed_docs = False
    # Store instances in session state to persist them
    if "doc_processor" not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor()
    if "retrieval_engine" not in st.session_state:
        st.session_state.retrieval_engine = RetrievalEngine()
    if "llm_builder" not in st.session_state:
        st.session_state.llm_builder = LLMChainBuilder()
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None

def main():
    st.set_page_config(AppConfig.APP_TITLE, page_icon=AppConfig.APP_ICON, layout=AppConfig.LAYOUT)
    st.title(AppConfig.APP_TITLE)
    
    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.header("📂 Document Center")
        st.markdown("Upload your research papers or documents here.")
        
        pdf_docs = st.file_uploader(
            "Upload PDFs",
            accept_multiple_files=True,
            type=['pdf']
        )

        if st.button("🚀 Process Documents", type="primary"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("⏳ Parsing, Chunking and Indexing..."):
                    # 1. Process PDFs
                    raw_docs = st.session_state.doc_processor.process_pdfs(pdf_docs)
                    
                    # 2. Chunk Documents
                    text_chunks = st.session_state.doc_processor.chunk_documents(raw_docs)
                    
                    # 3. Initialize/Update Vector Store (FAISS) & Hybrid Search
                    st.session_state.retrieval_engine.initialize_vector_store(text_chunks)
                    
                    # 4. Create Retriever (Hybrid + Reranking)
                    # Note: We need to handle the reranking integration. 
                    # For now, let's get the hybrid retriever.
                    # Ideally, we wrap this in a ContextualCompressionRetriever for automatic reranking in the chain.
                    
                    # Get base hybrid retriever
                    base_retriever = st.session_state.retrieval_engine.get_hybrid_retriever()
                    
                    # Set up Reranking (using LangChain's CrossEncoderReranker wrapper for convenience if possible, 
                    # otherwise using our custom logic via a custom retriever would be best, 
                    # but here we use the RetrievalEngine's reranker model path).
                    model = HuggingFaceCrossEncoder(model_name=ModelConfig.RERANKER_MODEL)
                    compressor = CrossEncoderReranker(model=model, top_n=5)
                    compression_retriever = ContextualCompressionRetriever(
                        base_compressor=compressor, base_retriever=base_retriever
                    )

                    # 5. Create LLM Chain
                    st.session_state.conversation_chain = st.session_state.llm_builder.create_chain(compression_retriever)
                    
                    st.session_state.processed_docs = True
                    st.success(f"✅ Indexed {len(raw_docs)} documents with Hybrid Search & Reranking!")
                    time.sleep(1)

        st.divider()
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.rerun()

    # Chat Interface
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 Source Citations"):
                    for i, doc in enumerate(msg["sources"]):
                        source = doc.metadata.get('source', 'Unknown')
                        page = doc.metadata.get('page', 'Unknown')
                        st.markdown(f"**{i+1}.** *{source}* (Page {page})")

    if user_question := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.conversation_chain:
             st.error("⚠️ Please upload and process documents first.")
        else:
            with st.chat_message("user"):
                st.markdown(user_question)
            
            # Add user message to state
            st.session_state.messages.append({"role": "user", "content": user_question})

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                # Run Chain
                response = st.session_state.conversation_chain({'question': user_question})
                answer_text = response.get('answer', "")
                source_docs = response.get('source_documents', [])

                # Stream response
                for char in stream_text(answer_text):
                    full_response += char
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)

                # Show Citations
                if source_docs:
                    with st.expander("📚 View Source Citations"):
                        for i, doc in enumerate(source_docs):
                            source = doc.metadata.get('source', 'Unknown')
                            page = doc.metadata.get('page', 'Unknown')
                            content_preview = doc.page_content[:200].replace("\n", " ") + "..."
                            st.markdown(f"**{i+1}.** *{source}* (Page {page})")
                            st.caption(content_preview)
            
            # Add assistant message to state
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": source_docs
            })

if __name__ == "__main__":
    main()