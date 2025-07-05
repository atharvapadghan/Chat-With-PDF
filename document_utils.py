import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

def process_uploaded_files(uploaded_files):
    """Process uploaded files and create documents"""
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    for uploaded_file in uploaded_files:
        try:
            content = ""
            
            # Handle different file types
            if uploaded_file.type == "text/plain":
                # Handle TXT files
                content = uploaded_file.read().decode("utf-8")
            elif uploaded_file.type == "application/pdf":
                # Handle PDF files
                try:
                    import PyPDF2
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    content = ""
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text.strip():  # Only add non-empty pages
                            content += page_text + "\n"
                except ImportError:
                    st.error("PyPDF2 is required for PDF processing. Please install it with: pip install PyPDF2")
                    continue
                except Exception as pdf_error:
                    st.error(f"Error processing PDF {uploaded_file.name}: {pdf_error}")
                    continue
            else:
                st.error(f"Unsupported file type: {uploaded_file.type}")
                continue
            
            # Check if content was extracted
            if not content.strip():
                st.warning(f"No text content found in {uploaded_file.name}")
                continue
            
            # Split text into chunks
            chunks = text_splitter.split_text(content)
            
            # Create documents
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={"source": uploaded_file.name, "chunk": i, "file_type": uploaded_file.type}
                )
                documents.append(doc)
                
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
    
    return documents

def create_vector_store(documents, embedding):
    """Create vector store from documents"""
    if not documents:
        return None, None
    
    try:
        # Create vector store
        vector_store = Chroma.from_documents(documents, embedding)
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4})
        return vector_store, retriever
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None, None 