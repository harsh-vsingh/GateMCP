# Dynamic LangGraph Assistant with HITL & MCP

A modular, production-ready AI assistant platform featuring a **Human-In-The-Loop (HITL)** approval system, local **RAG** capabilities, and dynamic **Model Context Protocol (MCP)** tool management. This project is optimized for local execution using Ollama on Arch Linux.

---

## 🚀 Overview

This project is built as a 3-tier application to provide a seamless interface for interacting with local LLMs while maintaining strict control over tool execution.

* **Frontend**: A Streamlit application providing a real-time chat interface and a management dashboard for tools.
* **Backend**: A FastAPI server that orchestrates the LangGraph state machine and serves as a gateway for tool orchestration.
* **Agent Layer**: A LangGraph workflow that handles decision-making, tool calling, and state persistence via PostgreSQL.

---

## ✨ Features

### 🛠️ Dynamic MCP Tool Management
Add or remove Model Context Protocol (MCP) servers at runtime. The backend automatically re-compiles the agent's logic to incorporate new tools (via Stdio or SSE) without requiring a process restart.

### 🛡️ Human-In-The-Loop (HITL)
Safety-first execution. The agent interrupts before calling sensitive tools, allowing the user to inspect, approve, or deny the action directly from the chat UI.

### 📚 Local RAG Pipeline
Ingest PDFs into a local Chroma vector database using `nomic-embed-text` embeddings. The agent can retrieve this information using the `query_knowledge_base` tool to provide context-aware answers.

### 💾 Persistent Conversations
Multi-threaded chat history is preserved across sessions using an `AsyncPostgresSaver` checkpointer, allowing you to resume any conversation by its `thread_id`.

---

## 📂 Project Structure

```text
chatbot/
├── backend/
│   ├── main.py          # FastAPI Entry Point & API Routes
│   ├── graph.py         # LangGraph Workflow & LLM Setup
│   ├── tools.py         # Built-in MCP Toolset (TLS)
│   ├── utils.py         # RAG, Streaming, & History Helpers
│   └── mp_service.py    # MCP Lifecycle & Validation Service
├── frontend/
│   └── app.py           # Streamlit Chat UI & Tool Manager
├── agent_workspace/     # Persistent Storage (Vectors & Configs)
│   ├── chroma_db/       # Vector Database files
│   └── mcp_servers/     # Individual MCP JSON configurations
├── .env                 # Environment Configuration
└── pyproject.toml       # Project Dependencies
```

## ⚙️ Setup & Installation

### 1. Prerequisites

Ensure the following are installed:

- **Ollama** (for local LLM + embeddings)
  - Pull model:
    ```bash
    ollama pull qwen3:8b
    ```
  - Pull embedding model:
    ```bash
    ollama pull nomic-embed-text
    ```

- **PostgreSQL**
  - Create database:
    ```sql
    CREATE DATABASE langgraph_db;
    ```

- **Python 3.12+**
  - Use `uv`

---

### 2. Configuration

Create a `.env` file in the root directory:

```env
DATABASE_URL=postgresql://user:password@localhost:5432/langgraph_db
TAVILY_API_KEY=your_tavily_key
BACKEND_URL=http://localhost:8000
LLM_MODEL=qwen3:8b
MODEL_URL=http://localhost:11434
```

### 3. Install dependencies
```bash
uv sync
```

---

## 🏃 Running the Application
``` bash
uv run uvicorn backend.main:app --reload
uv run streamlit run frontend/app.py
```

---

## 🛠️ Usage

### 💬 Chatting & HITL

- Enter a message in the chat input to start a conversation  
- When a sensitive tool is triggered:
  - The agent pauses and displays its intent  

**Actions:**
- Approve → continues execution  
- Deny → injects constraint and replans response  

---

### 📚 Local RAG

- Upload a PDF using the sidebar  
- Click **Process File**

System behavior:
- Splits document into chunks  
- Generates embeddings using `nomic-embed-text`  
- Stores vectors in local ChromaDB  

Querying:
- Use the `query_knowledge_base` tool  
- Responses are grounded in uploaded documents  

---

### 🛠️ MCP Tool Management

- Open **Manage MCP Servers** in the UI  
- Paste and save an MCP JSON configuration  

System behavior:
- Stores config in `agent_workspace/mcp_servers/`  
- Triggers automatic hot reload  
- Recompiles LangGraph with updated tools  

---

### 💾 Conversation Persistence

- Conversations are stored in PostgreSQL  
- Each session is tied to a `thread_id`  
- Enables resuming past conversations without loss of context  
