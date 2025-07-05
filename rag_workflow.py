import streamlit as st
from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.schema import Document
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver

# Define the data models
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

# Initialize LLM and embeddings
@st.cache_resource
def initialize_models():
    """Initialize and cache the LLM and embedding models"""
    try:
        llm = ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=0.5)
        embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        return llm, embedding
    except Exception as e:
        st.error(f"Error initializing models: {e}")
        return None, None

# Define the workflow functions
def question_rewriter(state: AgentState, llm):
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

def question_classifier(state: AgentState, llm):
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

def retrieval_grader(state: AgentState, llm):
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

def refine_question(state: AgentState, llm):
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

def generate_answer(state: AgentState, llm):
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
def create_workflow(llm):
    """Create and return the workflow graph"""
    checkpointer = MemorySaver()
    
    workflow = StateGraph(AgentState)
    
    # Add nodes with llm parameter
    workflow.add_node("question_rewriter", lambda state: question_rewriter(state, llm))
    workflow.add_node("question_classifier", lambda state: question_classifier(state, llm))
    workflow.add_node("off_topic_response", off_topic_response)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("retrieval_grader", lambda state: retrieval_grader(state, llm))
    workflow.add_node("generate_answer", lambda state: generate_answer(state, llm))
    workflow.add_node("refine_question", lambda state: refine_question(state, llm))
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