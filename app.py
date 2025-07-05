import streamlit as st
from rag_workflow import create_workflow, initialize_models
from document_utils import process_uploaded_files, create_vector_store
from ui_components import setup_page_config, load_custom_css, initialize_session_state
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# Setup page configuration and styling
setup_page_config()
load_custom_css()

# Initialize session state
initialize_session_state()

# Initialize LLM and embeddings
llm, embedding = initialize_models()
if llm is None or embedding is None:
    st.error("Failed to initialize models. Please check your API keys and internet connection.")
    st.stop()

# Main Streamlit app
def main():
    st.markdown('<div class="main-header">💬 Chat Interface</div>', unsafe_allow_html=True)
    
    # Sidebar for document upload
    with st.sidebar:
        st.markdown('<div class="sidebar-header">📁 Document Upload</div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            uploaded_files = st.file_uploader(
                "Upload your documents (TXT & PDF files)",
                type=['txt', 'pdf'],
                accept_multiple_files=True,
                help="Upload text files (TXT) or PDF files to create a knowledge base for the chatbot"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Display uploaded files info
        if uploaded_files:
            for file in uploaded_files:
                file_icon = "📄" if file.type == "text/plain" else "📕"
                st.markdown(f'<div class="file-info">{file_icon} {file.name}</div>', unsafe_allow_html=True)
        
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    # Process uploaded files
                    documents = process_uploaded_files(uploaded_files)
                    
                    if documents:
                        # Create vector store
                        vector_store, retriever = create_vector_store(documents, embedding)
                        
                        if vector_store and retriever:
                            st.session_state.vector_store = vector_store
                            st.session_state.retriever = retriever
                            st.session_state.documents_uploaded = True
                            st.session_state.docs = documents
                            
                            # Create workflow graph
                            st.session_state.graph = create_workflow(llm)
                            
                            st.success(f"✅ Documents processed!")
                        else:
                            st.error("❌ Failed to create vector store")
                    else:
                        st.error("❌ No documents were processed")
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            st.markdown(f"""
            <div class="chat-message">
                <span class="user-label">You:</span>
                <div class="user-message">{message.content}</div>
            </div>
            """, unsafe_allow_html=True)
        elif isinstance(message, AIMessage):
            st.markdown(f"""
            <div class="chat-message">
                <span class="assistant-label">Assistant:</span>
                <div class="assistant-message">{message.content}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input (text only)
    if st.session_state.documents_uploaded:
        prompt = st.chat_input("Ask a question about your documents...")
        if prompt:
            st.session_state.messages.append(HumanMessage(content=prompt))
            with st.spinner("Thinking..."):
                try:
                    input_data = {"question": HumanMessage(content=prompt)}
                    result = st.session_state.graph.invoke(
                        input=input_data, 
                        config={"configurable": {"thread_id": st.session_state.thread_id}}
                    )
                    st.session_state.messages = result["messages"]
                except Exception as e:
                    st.error(f"Error processing question: {e}")
                    st.session_state.messages.append(
                        AIMessage(content="Sorry, I encountered an error while processing your question.")
                    )
            st.rerun()
    else:
        st.info("👆 Please upload and process documents in the sidebar to start chatting!")

if __name__ == "__main__":
    main() 