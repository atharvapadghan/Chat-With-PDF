import streamlit as st
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_page_config():
    """Setup Streamlit page configuration"""
    st.set_page_config(
        page_title="Multi-RAG Conversational Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def load_custom_css():
    """Load custom CSS for modern dark chat UI"""
    st.markdown("""
    <style>
    body {
        background: #181820;
    }
    .main-header {
        font-size: 2.2rem;
        color: #fff;
        text-align: left;
        margin-bottom: 1.5rem;
        font-weight: 700;
        letter-spacing: -1px;
    }
    .chat-container {
        background: #23232b;
        border-radius: 18px;
        padding: 2rem 1.5rem 2.5rem 1.5rem;
        min-height: 60vh;
        max-width: 800px;
        margin: 0 auto;
        box-shadow: 0 2px 16px rgba(0,0,0,0.10);
    }
    .chat-message {
        margin-bottom: 1.5rem;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
    }
    .user-message {
        background: #e3f2fd;
        color: #222;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 0.2rem;
        width: fit-content;
        max-width: 90%;
        box-shadow: none;
    }
    .assistant-message {
        background: #f8e1f3;
        color: #222;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 0.2rem;
        width: fit-content;
        max-width: 90%;
        box-shadow: none;
    }
    .user-label {
        color: #2196f3;
        font-weight: 700;
        margin-bottom: 0.2rem;
        font-size: 1rem;
    }
    .assistant-label {
        color: #b13fae;
        font-weight: 700;
        margin-bottom: 0.2rem;
        font-size: 1rem;
    }
    .stTextInput>div>div>input {
        font-size: 1.1rem;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        background: #23232b;
        color: #fff;
        border: 1px solid #444;
    }
    .stTextInput>div>div>input:focus {
        border: 1.5px solid #2196f3;
    }
    .stChatInputContainer {
        background: #23232b !important;
        border-radius: 10px;
    }
    .sidebar-header {
        font-size: 1.3rem;
        color: #fff;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .upload-section {
        background-color: #23232b;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        color: #fff;
    }
    .file-info {
        background-color: #23232b;
        color: #fff;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        font-size: 0.95rem;
    }
    .stButton>button {
        font-size: 1.1rem;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        margin-top: 0.5rem;
        background: #2196f3;
        color: #fff;
        border: none;
        font-weight: 600;
    }
    .stButton>button:hover {
        background: #1769aa;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'documents_uploaded' not in st.session_state:
        st.session_state.documents_uploaded = False
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    if 'graph' not in st.session_state:
        st.session_state.graph = None
    if 'thread_id' not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if 'docs' not in st.session_state:
        st.session_state.docs = [] 