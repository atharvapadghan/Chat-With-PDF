import streamlit as st
import os
from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.schema import Document
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
import tempfile
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Multi-RAG Conversational Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark chat UI
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

# Initialize session state
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

# Initialize LLM and embeddings
@st.cache_resource
def initialize_models():
    try:
        llm = ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=0.5)
        embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        return llm, embedding
    except Exception as e:
        st.error(f"Error initializing models: {e}")
        return None, None

llm, embedding = initialize_models()
if llm is None or embedding is None:
    st.error("Failed to initialize models. Please check your API keys and internet connection.")
    st.stop()

# Define the data models and functions
class AgentState(TypedDict):
    messages: List[BaseMessage]
    documents: List[Document]
    on_topic: str
    rephrased_question: str
    proceed_to_generate: bool
    rephrase_count: int
    question: HumanMessage

class GradeQuestion(BaseModel):
    score: str = Field(
        description="Question is about the specified topics? If yes -> 'Yes' if not -> 'No'"
    )

class GradeDocument(BaseModel):
    score: str = Field(
        description="Document is relevant to the question? If yes -> 'Yes' if not -> 'No'"
    )

# Define the workflow functions
def question_rewriter(state: AgentState):
    """Rewrite the question to be more suitable for retrieval"""
    # Reset state variables except for 'question' and 'messages'
    state["documents"] = []
    state["on_topic"] = ""
    state["rephrased_question"] = ""
    state["proceed_to_generate"] = False
    state["rephrase_count"] = 0

    if "messages" not in state or state["messages"] is None:
        state["messages"] = []

    if state["question"] not in state["messages"]:
        state["messages"].append(state["question"])

    if len(state["messages"]) > 1:
        conversation = state["messages"][:-1]
        current_question = state["question"].content
        messages = [
            SystemMessage(
                content="You are a helpful assistant that rephrases the user's question to be a standalone question optimized for retrieval."
            )
        ]
        messages.extend(conversation)
        messages.append(HumanMessage(content=current_question))
        rephrase_prompt = ChatPromptTemplate.from_messages(messages)
        prompt = rephrase_prompt.format()
        response = llm.invoke(prompt)
        better_question = response.content.strip()
        state["rephrased_question"] = better_question
    else:
        state["rephrased_question"] = state["question"].content
    return state

def question_classifier(state: AgentState):
    """Classify if the question is on-topic"""
    # Get document context for classification
    doc_context = ""
    if st.session_state.get("docs", []):
        # Create a summary of available documents for context
        doc_sources = list(set([doc.metadata.get('source', 'Unknown') for doc in st.session_state["docs"]]))
        doc_context = f"Available documents: {', '.join(doc_sources)}"
    
    system_message = SystemMessage(
        content=f"""You are a classifier that determines whether a user's question is about the uploaded documents.

{doc_context}

If the question IS about the content in the uploaded documents, respond with 'Yes'. Otherwise, respond with 'No'.

Consider the question relevant if it asks about:
- Information contained in the uploaded documents
- Details, facts, or data from the documents
- Specific topics covered in the documents
- Requests for clarification about document content

Consider the question NOT relevant if it asks about:
- General knowledge not in the documents
- Personal opinions or advice
- Topics completely unrelated to the uploaded content
"""
    )

    human_message = HumanMessage(
        content=f"User question: {state['rephrased_question']}"
    )
    grade_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    structured_llm = llm.with_structured_output(GradeQuestion)
    grader_llm = grade_prompt | structured_llm
    result = grader_llm.invoke({})
    state["on_topic"] = result.score.strip()
    return state

def on_topic_router(state: AgentState):
    """Route based on whether the question is on-topic"""
    on_topic = state.get("on_topic", "").strip().lower()
    if on_topic == "yes":
        return "retrieve"
    else:
        return "off_topic_response"

def retrieve(state: AgentState):
    """Retrieve relevant documents"""
    if st.session_state.retriever is None:
        state["documents"] = []
        state["proceed_to_generate"] = False
        return state
    
    documents = st.session_state.retriever.invoke(state["rephrased_question"])
    state["documents"] = documents
    return state

def retrieval_grader(state: AgentState):
    """Grade the relevance of retrieved documents"""
    if not state["documents"]:
        state["proceed_to_generate"] = False
        return state
    
    system_message = SystemMessage(
        content="""You are a grader assessing the relevance of a retrieved document to a user question.
Only answer with 'Yes' or 'No'.

If the document contains information relevant to the user's question, respond with 'Yes'.
Otherwise, respond with 'No'."""
    )

    structured_llm = llm.with_structured_output(GradeDocument)

    relevant_docs = []
    for doc in state["documents"]:
        human_message = HumanMessage(
            content=f"User question: {state['rephrased_question']}\\n\\nRetrieved document:\\n{doc.page_content}"
        )
        grade_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        grader_llm = grade_prompt | structured_llm
        result = grader_llm.invoke({})
        if result.score.strip().lower() == "yes":
            relevant_docs.append(doc)
    
    state["documents"] = relevant_docs
    state["proceed_to_generate"] = len(relevant_docs) > 0
    return state

def proceed_router(state: AgentState):
    """Route based on whether to proceed with generation"""
    rephrase_count = state.get("rephrase_count", 0)
    if state.get("proceed_to_generate", False):
        return "generate_answer"
    elif rephrase_count >= 2:
        return "cannot_answer"
    else:
        return "refine_question"

def refine_question(state: AgentState):
    """Refine the question to improve retrieval"""
    rephrase_count = state.get("rephrase_count", 0)
    if rephrase_count >= 2:
        return state
    
    question_to_refine = state["rephrased_question"]
    system_message = SystemMessage(
        content="""You are a helpful assistant that slightly refines the user's question to improve retrieval results.
Provide a slightly adjusted version of the question."""
    )
    human_message = HumanMessage(
        content=f"Original question: {question_to_refine}\\n\\nProvide a slightly refined question."
    )
    refine_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    prompt = refine_prompt.format()
    response = llm.invoke(prompt)
    refined_question = response.content.strip()
    state["rephrased_question"] = refined_question
    state["rephrase_count"] = rephrase_count + 1
    return state

def generate_answer(state: AgentState):
    """Generate the final answer"""
    if "messages" not in state or state["messages"] is None:
        raise ValueError("State must include 'messages' before generating an answer.")

    history = state["messages"]
    documents = state["documents"]
    rephrased_question = state["rephrased_question"]

    # Create RAG chain
    template = """Answer the question based on the following context and the Chat history. Especially take the latest question into consideration:

Chat history: {history}

Context: {context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = prompt | llm

    response = rag_chain.invoke(
        {"history": history, "context": documents, "question": rephrased_question}
    )

    generation = response.content.strip()
    state["messages"].append(AIMessage(content=generation))
    return state

def cannot_answer(state: AgentState):
    """Handle cases where we cannot answer"""
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []
    state["messages"].append(
        AIMessage(
            content="I'm sorry, but I cannot find the information you're looking for in the uploaded documents."
        )
    )
    return state

def off_topic_response(state: AgentState):
    """Handle off-topic questions"""
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []
    state["messages"].append(
        AIMessage(
            content="I'm sorry! I can only answer questions about the documents you've uploaded. Please ask a question related to the uploaded content."
        )
    )
    return state

# Create the workflow graph
def create_workflow():
    """Create and return the workflow graph"""
    checkpointer = MemorySaver()
    
    workflow = StateGraph(AgentState)
    workflow.add_node("question_rewriter", question_rewriter)
    workflow.add_node("question_classifier", question_classifier)
    workflow.add_node("off_topic_response", off_topic_response)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("retrieval_grader", retrieval_grader)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("refine_question", refine_question)
    workflow.add_node("cannot_answer", cannot_answer)

    workflow.add_edge("question_rewriter", "question_classifier")
    workflow.add_conditional_edges(
        "question_classifier",
        on_topic_router,
        {
            "retrieve": "retrieve",
            "off_topic_response": "off_topic_response",
        },
    )
    workflow.add_edge("retrieve", "retrieval_grader")
    workflow.add_conditional_edges(
        "retrieval_grader",
        proceed_router,
        {
            "generate_answer": "generate_answer",
            "refine_question": "refine_question",
            "cannot_answer": "cannot_answer",
        },
    )
    workflow.add_edge("refine_question", "retrieve")
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("cannot_answer", END)
    workflow.add_edge("off_topic_response", END)
    workflow.set_entry_point("question_rewriter")
    
    return workflow.compile(checkpointer=checkpointer)

# Document processing functions
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

def create_vector_store(documents):
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
                        vector_store, retriever = create_vector_store(documents)
                        
                        if vector_store and retriever:
                            st.session_state.vector_store = vector_store
                            st.session_state.retriever = retriever
                            st.session_state.documents_uploaded = True
                            st.session_state.docs = documents
                            
                            # Create workflow graph
                            st.session_state.graph = create_workflow()
                            
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