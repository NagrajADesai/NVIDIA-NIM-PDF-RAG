import streamlit as st
import time
from src.utils import get_pdf_documents, get_document_chunks, get_vector_store, get_conversational_chain

def stream_text(text):
    """Yields text one character at a time for streaming effect."""
    for char in text:
        yield char
        time.sleep(0.005)

def user_input(user_question):
    if "conversation" not in st.session_state or st.session_state.conversation is None:
        try:
            vector_store = get_vector_store()
            st.session_state.conversation = get_conversational_chain(vector_store)
        except Exception:
            st.error("⚠️ Please upload a PDF first to initialize the knowledge base.")
            return

    # Create a placeholder for the assistant's response to stream into
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Call the chain. Note: this chain currently doesn't support true streaming events easily
        # without callback handlers, so we'll simulate streaming for the final text.
        # Alternatively, for true streaming we'd need AsyncCallbackHandler.
        # For this "simple" personal project request, simulated streaming is often sufficient and easier.
        response = st.session_state.conversation({'question': user_question})
        
        answer_text = response.get('answer', "")
        source_docs = response.get('source_documents', [])

        # Simulate streaming output
        for char in stream_text(answer_text):
            full_response += char
            message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
        
        # Display Citations
        if source_docs:
            with st.expander("📚 View Source Citations"):
                for i, doc in enumerate(source_docs):
                    source = doc.metadata.get('source', 'Unknown')
                    page = doc.metadata.get('page', 'Unknown')
                    content_preview = doc.page_content[:200].replace("\n", " ") + "..."
                    st.markdown(f"**{i+1}.** *{source}* (Page {page})")
                    st.caption(content_preview)

    # Update session state with the new interaction
    st.session_state.messages.append({"role": "user", "content": user_question})
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response,
        "sources": source_docs # Store sources to potentially re-render if needed
    })


def main():
    st.set_page_config("NVIDIA NIM RAG", page_icon="🧠", layout="wide")
    
    st.title("🧠 NVIDIA NIM PDF Chat")
    
    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None 
    if "messages" not in st.session_state:
        st.session_state.messages = [] 

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
                with st.spinner("⏳ Parsing and Indexing..."):
                    # Use new utility functions
                    raw_docs = get_pdf_documents(pdf_docs)
                    text_chunks = get_document_chunks(raw_docs)
                    
                    # Create/Update Vector Store
                    vector_store = get_vector_store(text_chunks=text_chunks)
                    
                    # Re-initialize chain
                    st.session_state.conversation = get_conversational_chain(vector_store)
                    
                    st.success(f"✅ Indexed {len(raw_docs)} documents!")
                    time.sleep(1) # feedback delay
        
        st.divider()
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.rerun()

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Re-display sources if available in history
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 Source Citations"):
                    for i, doc in enumerate(msg["sources"]):
                        source = doc.metadata.get('source', 'Unknown')
                        page = doc.metadata.get('page', 'Unknown')
                        st.markdown(f"**{i+1}.** *{source}* (Page {page})")

    # Chat Input
    if user_question := st.chat_input("Ask a question about your documents..."):
        with st.chat_message("user"):
            st.markdown(user_question)
        
        user_input(user_question)

if __name__ == "__main__":
    main()