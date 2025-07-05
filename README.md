# 🤖 Multi-RAG Conversational Chatbot

A sophisticated conversational chatbot built with Streamlit, LangChain, and LangGraph that allows you to upload documents and have intelligent conversations about their content.

## ✨ Features

- **📄 Document Upload**: Upload TXT and PDF files to create your knowledge base
- **🔍 Intelligent Retrieval**: Advanced RAG (Retrieval-Augmented Generation) with MMR search
- **💬 Conversational Memory**: Maintains context across multiple turns
- **🎯 Question Refinement**: Automatically refines questions for better retrieval
- **📊 Relevance Grading**: Evaluates document relevance before generating answers
- **🔄 Multi-Step Reasoning**: Uses LangGraph for sophisticated workflow management
- **🎨 Modern UI**: Beautiful, responsive Streamlit interface with dark theme

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd multi-rag-chatbot
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a `.env` file in the project root:
```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### 4. Run the Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 How to Use

### 1. Upload Documents
- Use the sidebar to upload TXT or PDF files
- The system will process and chunk your documents
- You'll see a success message when processing is complete

### 2. Start Chatting
- Type your questions in the chat interface
- The chatbot will:
  - Rewrite your question for better retrieval
  - Classify if the question is relevant to your documents
  - Retrieve relevant document chunks
  - Grade the relevance of retrieved documents
  - Generate comprehensive answers

### 3. Features
- **Clear Chat**: Use the "Clear Chat History" button to start fresh
- **Conversational Memory**: The bot remembers previous context
- **Off-topic Detection**: The bot will tell you if your question isn't about the uploaded documents

## 🏗️ Architecture

The application uses a sophisticated multi-step workflow:

1. **Question Rewriter**: Optimizes questions for retrieval
2. **Question Classifier**: Determines if the question is on-topic
3. **Document Retriever**: Finds relevant document chunks
4. **Relevance Grader**: Evaluates document relevance
5. **Answer Generator**: Creates comprehensive responses

## 🔧 Technical Details

### Dependencies
- **Streamlit**: Web interface
- **LangChain**: RAG framework
- **LangGraph**: Workflow orchestration
- **ChromaDB**: Vector database
- **Sentence Transformers**: Text embeddings
- **Anthropic Claude**: Language model
- **PyPDF2**: PDF file processing

### Modular File Structure
```
multi-rag-chatbot/
├── app.py              # Main Streamlit application and UI
├── rag_workflow.py     # LangGraph workflow and RAG logic
├── document_utils.py   # Document processing and vector store creation
├── ui_components.py    # UI configuration, CSS, and session state
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── .env                # Environment variables (create this)
```

### Module Responsibilities

- **`app.py`**: Main application entry point, handles user interactions and chat interface
- **`rag_workflow.py`**: Contains the LangGraph workflow definition and model initialization
- **`document_utils.py`**: Handles document processing, text extraction, and vector store creation
- **`ui_components.py`**: Manages UI styling, page configuration, and session state initialization

## 🎯 Example Usage

1. **Upload any text documents** (company policies, research papers, manuals, etc.) or **PDF files**
2. **Ask questions** about the content:
   - "What are the main topics covered?"
   - "Can you summarize the key points?"
   - "What are the important dates mentioned?"

3. **Have a conversation**:
   - User: "What is this document about?"
   - Bot: "Based on the document, it covers..."
   - User: "Can you tell me more about [specific topic]?"
   - Bot: "According to the document, [detailed explanation]..."

## 🛠️ Customization

### Adding New File Types
To support additional file formats, modify the `process_uploaded_files` function in `document_utils.py`.

### Changing the Language Model
Update the `model_name` parameter in the `ChatAnthropic` initialization in `rag_workflow.py`.

### Adjusting Retrieval Parameters
Modify the `search_kwargs` in the retriever configuration in `document_utils.py`.

### Modifying the UI
Update the CSS styles and UI components in `ui_components.py`.

## 🐛 Troubleshooting

### Common Issues

1. **API Key Error**: Make sure your `.env` file contains the correct Anthropic API key
2. **Memory Issues**: For large documents, consider reducing chunk size in `document_utils.py`
3. **Slow Performance**: The first run may be slow due to model downloads
4. **PDF Processing Error**: Ensure PyPDF2 is installed: `pip install PyPDF2`

### Getting Help
- Check that all dependencies are installed correctly
- Ensure your API key is valid and has sufficient credits
- Try with smaller documents first

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Happy Chatting! 🤖💬** 

## RAG Workflow Structure

The following diagram illustrates the high-level workflow of the Retrieval-Augmented Generation (RAG) system implemented in this project:

![RAG Workflow Diagram]([ChatWithPdf.png](https://github.com/atharvapadghan/Chat-With-PDF/issues/1#issue-3204980907))

**Diagram Node to Code Mapping:**

- **question_rewriter**:  
  Handled in `rag_workflow.py` – rewrites or reformulates user questions for better retrieval.
- **question_classifier**:  
  Implemented in `rag_workflow.py` – classifies questions to determine if they are answerable, off-topic, or need refinement.
- **retrieve**:  
  Managed by `document_utils.py` and `rag_workflow.py` – retrieves relevant documents from the vector store (ChromaDB) using HuggingFace embeddings.
- **retrieval_grader**:  
  Logic in `rag_workflow.py` – evaluates the quality of retrieved documents and decides the next step.
- **refine_question**:  
  If needed, handled in `rag_workflow.py` – asks the user for clarification or reformulates the question.
- **generate_answer**:  
  Uses Anthropic Claude (via `rag_workflow.py`) to generate a final answer based on retrieved context.
- **cannot_answer**:  
  Returns a fallback response if the system cannot answer (handled in `rag_workflow.py`).
- **off_topic_response**:  
  Returns a response for off-topic questions (handled in `rag_workflow.py`).

**UI Integration:**  
The user interface in `app.py` and `ui_components.py` interacts with this workflow, allowing users to upload documents, ask questions, and view responses in a modern chat interface.

---

**Note:**
- Please save your workflow diagram as `rag_workflow.png` in a new `docs/` folder at your project root for the image to render correctly in the README. 
