import streamlit as st
from src.utils import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain


def user_input(user_question):
    if "conversation" not in st.session_state or st.session_state.conversation is None:
        try:
            # Load existing vector DB
            vector_store = get_vector_store()
            st.session_state.conversation = get_conversational_chain(vector_store)
        except Exception as e:
            st.error(f"❌ Failed to initialize conversation: {e}")
            return

    # Proceed with user question
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']



def main():
    st.set_page_config("LLM PDF Reader")
    st.header("🧠 LLM PDF Chat")

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None 
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None 
    if "messages" not in st.session_state:
        st.session_state.messages = []  # for interactive chat display

    # Display previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input (like ChatGPT)
    user_question = st.chat_input("Ask a Question from the PDF files")

    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)

        # Call user_input function
        user_input(user_question)

        # Get the last assistant reply from chatHistory
        if st.session_state.chatHistory:
            reply = st.session_state.chatHistory[-1].content
            with st.chat_message("assistant"):
                st.markdown(reply)

            # Store messages in session for rendering
            st.session_state.messages.append({"role": "user", "content": user_question})
            st.session_state.messages.append({"role": "assistant", "content": reply})

    # Sidebar for uploading PDFs
    with st.sidebar:
        st.title("📚 Menu")
        pdf_docs = st.file_uploader(
            "Upload your PDF files and click on the Submit & Process Button",
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks=text_chunks)
                st.session_state.conversation = get_conversational_chain(vector_store)

                st.success("✅ Done. Ask your questions in the main window.")
                st.session_state.messages = []  # reset chat on new PDF load




if __name__ == "__main__":
    main()