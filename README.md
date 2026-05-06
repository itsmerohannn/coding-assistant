# AI Coding Assistant Agent

An agentic AI assistant capable of explaining, reviewing, and improving Python code using **Retrieval-Augmented Generation (RAG)** and state-of-the-art **Transformer-based LLMs**. This tool is designed to provide context-aware responses by indexing local documentation or code snippets into a vector database.

## Key Features

*   **Three Operational Modes**:
    *   **Explain**: Generates concise, high-level summaries of complex logic.
    *   **Review**: Identifies logical bugs, PEP 8 violations, and performance bottlenecks.
    *   **Improve**: Refactors code for efficiency and readability while maintaining functionality.
*   **Context-Aware Reasoning**: Uses a RAG pipeline to pull relevant information from a local knowledge base (indexed via FAISS).
*   **Modular Architecture**: Separated concerns for model inference, vector retrieval, and agent workflows.
*   **Real-time Interface**: Built with Streamlit for an interactive developer experience.

## Technical Stack

*   **LLM**: `Qwen2.5-Coder-1.5B-Instruct` (Optimized for code reasoning).
*   **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`.
*   **Vector Store**: **FAISS** (Facebook AI Similarity Search).
*   **Backend**: Python / PyTorch.
*   **Frontend**: Streamlit.

## Architecture

The system follows an **Agentic Workflow**:
1.  **Ingestion**: User documentation is chunked and embedded into a high-dimensional vector space.
2.  **Retrieval**: When a query is made, the system performs a similarity search in the FAISS index to find relevant context.
3.  **Augmentation**: The prompt is dynamically constructed using the retrieved context, the target code, and specific task instructions.
4.  **Generation**: The LLM processes the augmented prompt to produce a grounded, non-hallucinatory response.



---

## Getting Started

### Prerequisites
*   Python 3.10+
*   16GB RAM (Recommended for local inference)

### Installation

1. **Clone the repository**:
   
   ```
   git clone [https://github.com/itsmerohannn/coding-assistant](https://github.com/itsmerohannn/coding-assistant)
   cd coding-assistant
   ```
3. **Create a virtual environment**:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. **Install dependencies**:
   `pip install -r requirements.txt`

### Usage
Run the Streamlit application:
   ```
  streamlit run app.py
   ```
  1. Sidebar: Paste your reference documentation into the "Knowledge Base" section.

  2. Main Panel: Paste your Python code and select a task (Explain, Review, or Improve).

  3. Run: Click "Run Assistant" to see the agent's analysis.

### Project Strcuture
```
├── app.py              # Streamlit UI and application entry point
├── src/
│   ├── assistant.py    # Agent logic and prompt engineering
│   ├── models.py       # LLM loading and inference utilities
│   ├── rag.py          # Vector indexing and retrieval logic
│   └── config.py       # Model identifiers and hyperparameters
├── tests/              # Unit tests for agent workflows
└── requirements.txt    # Project dependencies
```

**Author**
**Vamsi Rohan Tamadala**
Student at GITAM University | Aspiring ML Engineer
