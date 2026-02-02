import streamlit as st
import nest_asyncio
nest_asyncio.apply()
import time
from src.config import AppConfig
from src.document_processor import DocumentProcessor
from src.retrieval_engine import RetrievalEngine
from src.vector_manager import VectorStoreManager

def main():
    st.set_page_config("Create Knowledgebase", page_icon="📂", layout="wide")
    st.title("📂 Knowledgebase Manager")

    doc_processor = DocumentProcessor()
    retrieval_engine = RetrievalEngine()
    vector_manager = VectorStoreManager()

    # Sidebar
    dbs = vector_manager.list_dbs()
    with st.sidebar:
        st.header("Existing Databases")
        if dbs:
            st.markdown("\n".join([f"- {db}" for db in dbs]))
        else:
            st.info("No databases found.")

    # Main Area
    st.subheader("🆕 Create or Update a Knowledgebase")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        db_name = st.text_input("Database Name", placeholder="e.g., Finance_Reports_2024")
        
        uploaded_files = st.file_uploader(
            "Upload Documents",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'pptx', 'xlsx', 'txt']
        )
        
        if st.button("🚀 Process & Create/Update", type="primary"):
            if not db_name:
                st.error("⚠️ Please specify a Database Name.")
            elif not uploaded_files:
                st.error("⚠️ Please upload at least one file.")
            else:
                 # Sanitize DB name roughly
                 safe_name = "".join([c for c in db_name if c.isalnum() or c in ('_', '-')])
                 
                 with st.spinner(f"⏳ Processing into '{safe_name}'..."):
                     try:
                        # 1. Process Files
                        raw_docs = doc_processor.process_files(uploaded_files)
                        
                        # Show Data Engineering Logs
                        with st.status("🛠️ Data Engineering Log", expanded=True):
                            for log in doc_processor.logger.logs:
                                if log['step'] == 'Error':
                                    st.error(f"**{log['file']}**: {log['details']}")
                                elif log['step'] == 'Warning':
                                    st.warning(f"**{log['file']}**: {log['details']}")
                                else:
                                    st.write(f"**[{log['step']}]** {log['file']}: {log['details']}")
                        
                        if not raw_docs:
                            st.warning("No valid text extracted from uploaded files.")
                        else:
                            # 2. Chunk Documents
                            text_chunks = doc_processor.chunk_documents(raw_docs)
                            st.info(f"Generated {len(text_chunks)} chunks from {len(raw_docs)} extracted segments.")
                            
                            # 3. Get Path
                            db_path = vector_manager.create_db_dir(safe_name)
                            
                            # 4. Initialize/Update Vector Store
                            retrieval_engine.initialize_vector_store(text_chunks, save_path=db_path)
                            
                            st.success(f"✅ Successfully updated knowledgebase: **{safe_name}**")
                            time.sleep(1)
                            st.rerun()
                     except Exception as e:
                         st.error(f"❌ Error: {str(e)}")

    with col2:
         st.warning("⚠️ **Note**: Updating an existing database with the same name will merge new documents into it.")
         st.markdown("""
         ### Supported Formats:
         - **PDF** (.pdf)
         - **Word** (.docx)
         - **PowerPoint** (.pptx)
         - **Excel** (.xlsx)
         - **Text** (.txt)
         """)

if __name__ == "__main__":
    main()
