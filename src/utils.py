import os
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

load_dotenv(dotenv_path=".env")

NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY')
os.environ['NVIDIA_API_KEY'] = NVIDIA_API_KEY

MODEL = "meta/llama-3.1-8b-instruct"

def get_pdf_documents(pdf_docs):
    """
    Reads PDF files using PyMuPDF and returns a list of LangChain Document objects
    with metadata (source filename and page number).
    """
    documents = []
    for pdf in pdf_docs:
        # Save uploaded file temporarily to read with fitz (or read from stream if possible, 
        # but fitz often needs a file or bytes)
        # Streamlit UploadedFile is a BytesIO-like object.
        pdf_bytes = pdf.read()
        
        # Open with PyMuPDF
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text.strip():  # Skip empty pages
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "source": pdf.name,
                            "page": page_num + 1
                        }
                    ))
    return documents

def get_document_chunks(documents):
    """
    Splits a list of Documents into smaller chunks while preserving metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    return chunks

def get_vector_store(text_chunks=None, db_path="faiss_db", model_path="all-MiniLM-L6-v2"):
    # Load embedding model from local path
    embeddings = HuggingFaceEmbeddings(model_name=model_path)
    
    # Check if FAISS DB already exists
    db_exists = os.path.exists(os.path.join(db_path, "index.faiss")) and \
                os.path.exists(os.path.join(db_path, "index.pkl"))

    if db_exists:
        # Load existing DB
        vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        # print("✅ Loaded existing vector DB.")
    else:
        vector_store = None
        # print("❌ No existing vector DB found.")

    # If new text chunks are given, update or create DB
    if text_chunks:
        # print("📄 Adding new documents to vector DB...")
        if vector_store:
            # Add new documents to existing DB
            new_store = FAISS.from_documents(text_chunks, embedding=embeddings)
            vector_store.merge_from(new_store)
        else:
            # Create new DB from scratch
            vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

        # Save updated DB
        vector_store.save_local(db_path)
        # print(f"💾 Vector DB saved to {db_path}")

    # If still no vector_store (e.g., no DB and no new docs)
    if not vector_store:
        raise ValueError("No documents provided and no existing vector DB found.")

    return vector_store


def get_conversational_chain(vector_store):
    llm = ChatNVIDIA(
        base_url = "https://integrate.api.nvidia.com/v1",
        model_name=MODEL,
        temperature=0.3,
        top_p=0.7,
        max_tokens=1024,
    )    

    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="answer" # Neccessary when return_source_documents=True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=vector_store.as_retriever(),
        memory=memory,
        return_source_documents=True, # Enable citations
        verbose=False
    )
    
    return conversation_chain