import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv(dotenv_path=".env")

NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY')
os.environ['NVIDIA_API_KEY'] = NVIDIA_API_KEY


MODEL = "meta/llama-3.1-8b-instruct"

# extract text from the pdf files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# create chunks of text
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    
    return chunks

## using google embedding model
# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    
#     return vector_store

## using miniLM embedding model (open-source)
# def get_vector_store(text_chunks):
#     embeddings = HuggingFaceEmbeddings(model_name="D:/mpcl/models/all-MiniLM-L6-v2")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     return vector_store


# change the db_path and embedding model_path accordingly
def get_vector_store(text_chunks=None, db_path="faiss_db", model_path="all-MiniLM-L6-v2"):
    # Load embedding model from local path
    embeddings = HuggingFaceEmbeddings(model_name=model_path)
    
    # Check if FAISS DB already exists
    db_exists = os.path.exists(os.path.join(db_path, "index.faiss")) and \
                os.path.exists(os.path.join(db_path, "index.pkl"))

    if db_exists:
        # Load existing DB
        vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        print("✅ Loaded existing vector DB.")
    else:
        vector_store = None
        print("❌ No existing vector DB found.")

    # If new text chunks are given, update or create DB
    if text_chunks:
        print("📄 Adding new documents to vector DB...")
        if vector_store:
            # Add new documents to existing DB
            new_store = FAISS.from_texts(text_chunks, embedding=embeddings)
            vector_store.merge_from(new_store)
        else:
            # Create new DB from scratch
            vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

        # Save updated DB
        vector_store.save_local(db_path)
        print(f"💾 Vector DB saved to {db_path}")

    # If still no vector_store (e.g., no DB and no new docs)
    if not vector_store:
        raise ValueError("No documents provided and no existing vector DB found.")

    return vector_store


def get_conversational_chain(vector_store):

    # openai model
    llm = ChatNVIDIA(
        base_url = "https://integrate.api.nvidia.com/v1",
        model_name=MODEL,
        temperature=0.3,
        top_p=0.7,
        max_tokens=1024,
    )    

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=vector_store.as_retriever(),
        memory=memory,
        verbose=False
    )
    
    return conversation_chain