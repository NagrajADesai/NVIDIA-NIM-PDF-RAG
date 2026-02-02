import streamlit as st
import nest_asyncio
nest_asyncio.apply()
import time
from src.config import AppConfig, ModelConfig
from src.retrieval_engine import RetrievalEngine
from src.vector_manager import VectorStoreManager
from src.agent_graph import build_graph

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
    if "agent_app" not in st.session_state:
        st.session_state.agent_app = None

def load_agent(db_name, vector_manager, retrieval_engine):
    """Loads the agent for the selected DB."""
    try:
        db_path = vector_manager.get_db_path(db_name)
        
        # Initialize Vector Store/Retriever components (Load existing)
        retrieval_engine.initialize_vector_store(text_chunks=None, save_path=db_path)
        
        # Build Hybrid Retriever
        retriever = retrieval_engine.get_hybrid_retriever()
        
        # Build Graph
        return build_graph(retriever)
    except Exception as e:
        st.error(f"Error loading database '{db_name}': {e}")
        return None

def main():
    st.set_page_config("Chat With Data", page_icon="💬", layout="wide")
    st.title("💬 Chat With Agent")

    initialize_chat_state()
    
    vector_manager = VectorStoreManager()
    retrieval_engine = RetrievalEngine()

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
            with st.spinner(f"Loading Agent for '{selected_db}'..."):
                 st.session_state.agent_app = load_agent(selected_db, vector_manager, retrieval_engine)
    
    # Check Agent Availability
    if not st.session_state.agent_app:
        if not dbs:
             st.info("👈 Please create a Knowledgebase first.")
        else:
             pass
    else:
        # Chat Interface
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
                # Show partial steps if preserved (optional, kept simple for history)
                if "steps" in msg and msg["steps"]:
                    with st.expander("🧠 Agent Thoughts (History)"):
                        for step in msg["steps"]:
                            st.write(f"- {step}")

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
                steps_display = st.status("🧠 Agent Thinking...", expanded=True)
                
                try:
                    # Invoke Agent
                    inputs = {"question": user_question}
                    final_state = st.session_state.agent_app.invoke(inputs)
                    
                    answer_text = final_state.get("generation", "I couldn't generate an answer.")
                    source_docs = final_state.get("documents", [])
                    steps = final_state.get("steps", [])
                    
                    # Update status with steps
                    for step in steps:
                         steps_display.write(f"- {step}")
                    steps_display.update(label="🧠 Agent Finished Thinking", state="complete", expanded=False)

                    # Stream Response
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
                        "sources": source_docs,
                        "steps": steps
                    })
                except Exception as e:
                    steps_display.update(label="❌ Error", state="error")
                    st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()

