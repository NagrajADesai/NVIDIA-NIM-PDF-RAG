import streamlit as st
import time
from src.config import AppConfig, ModelConfig
from src.retrieval_engine import RetrievalEngine
from src.llm_chain import LLMChainBuilder
from src.vector_manager import VectorStoreManager
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

def stream_text(text):
    """Yields text one character at a time for streaming effect."""
    for char in text:
        yield char
        time.sleep(0.005)

def initialize_chat_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_db" not in st.session_state:
        st.session_state.current_db = None
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None

def load_chain(db_name, vector_manager, retrieval_engine, llm_builder):
    """Loads the chain for the selected DB."""
    try:
        db_path = vector_manager.get_db_path(db_name)
        
        # Initialize Vector Store for this DB (No new chunks, just load)
        retrieval_engine.initialize_vector_store(text_chunks=None, save_path=db_path)
        
        # Build Retriever
        base_retriever = retrieval_engine.get_hybrid_retriever()
        
        # Reranker
        model = HuggingFaceCrossEncoder(model_name=ModelConfig.RERANKER_MODEL)
        compressor = CrossEncoderReranker(model=model, top_n=5)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )
        
        # Chain
        return llm_builder.create_chain(compression_retriever)
    except Exception as e:
        st.error(f"Error loading database '{db_name}': {e}")
        return None

def main():
    st.set_page_config("Chat With Data", page_icon="💬", layout="wide")
    st.title("💬 Chat With Data")

    initialize_chat_state()
    
    vector_manager = VectorStoreManager()
    # Note: We instantiate these fresh for the page re-run or use session state if expensive to init
    # Since these classes are light-weight (heavy lifting in methods), re-init is fine.
    retrieval_engine = RetrievalEngine()
    llm_builder = LLMChainBuilder()

    dbs = vector_manager.list_dbs()

    # Sidebar: DB Selection
    with st.sidebar:
        st.header("⚙️ Configuration")
        if not dbs:
            st.warning("No Knowledgebases found. Please create one in the 'Creating Knowledgebase' page.")
            selected_db = None
        else:
            selected_db = st.selectbox(
                "Select Knowledgebase", 
                options=dbs,
                index=dbs.index(st.session_state.current_db) if st.session_state.current_db in dbs else 0
            )

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Handle DB Switch
    if selected_db:
        if selected_db != st.session_state.current_db:
            st.session_state.current_db = selected_db
            st.session_state.messages = [] # Clear history on switch
            with st.spinner(f"Loading '{selected_db}'..."):
                 st.session_state.conversation_chain = load_chain(selected_db, vector_manager, retrieval_engine, llm_builder)
    
    # Check Chain Availability
    if not st.session_state.conversation_chain:
        if not dbs:
             st.info("👈 Please create a Knowledgebase first.")
        else:
             # Should stick here if load failed or waiting for selection
             pass
    else:
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

        if user_question := st.chat_input("Ask a question..."):
            with st.chat_message("user"):
                st.markdown(user_question)
            
            st.session_state.messages.append({"role": "user", "content": user_question})

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                try:
                    response = st.session_state.conversation_chain({'question': user_question})
                    answer_text = response.get('answer', "")
                    source_docs = response.get('source_documents', [])

                    for char in stream_text(answer_text):
                        full_response += char
                        message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)

                    if source_docs:
                        with st.expander("📚 View Source Citations"):
                            for i, doc in enumerate(source_docs):
                                source = doc.metadata.get('source', 'Unknown')
                                page = doc.metadata.get('page', 'Unknown')
                                content_preview = doc.page_content[:200].replace("\n", " ") + "..."
                                st.markdown(f"**{i+1}.** *{source}* (Page {page})")
                                st.caption(content_preview)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": source_docs
                    })
                except Exception as e:
                    st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()
