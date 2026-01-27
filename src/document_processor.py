import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List
from src.config import AppConfig

class DocumentProcessor:
    """Handles document ingestion and processing."""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=AppConfig.CHUNK_SIZE,
            chunk_overlap=AppConfig.CHUNK_OVERLAP
        )

    def process_pdfs(self, pdf_files) -> List[Document]:
        """
        Reads PDF files and returns a list of LangChain Document objects.
        
        Args:
            pdf_files: List of uploaded PDF files (file-like objects).
            
        Returns:
            List[Document]: List of documents with content and metadata.
        """
        documents = []
        for pdf in pdf_files:
            try:
                # Handle Streamlit UploadedFile (read bytes)
                pdf_bytes = pdf.read()
                
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
            except Exception as e:
                print(f"Error processing {pdf.name}: {e}")
        
        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits documents into smaller chunks.
        
        Args:
            documents: List of Document objects.
            
        Returns:
            List[Document]: List of chunked Document objects.
        """
        return self.text_splitter.split_documents(documents)
