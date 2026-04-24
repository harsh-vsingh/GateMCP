# GateGraph: Human-Gated Agent with Dynamic MCP Registry

A locally hosted AI agent system that enforces **human-controlled tool execution** and supports **runtime MCP plugin registration**. Designed as a control layer over LLM agents, enabling safe, extensible, and stateful interactions.

---

## 🚀 Overview

This project implements a **human-gated agent architecture** where all external tool executions are explicitly controlled, and capabilities can be extended at runtime via a dynamic MCP registry.

Unlike standard agent pipelines, this system introduces a **control layer** between decision-making and execution, ensuring safe and auditable interactions.

The system is structured as a 3-tier architecture:

- **Frontend**: Streamlit interface for chat, HITL approvals, and MCP tool management  
- **Backend**: FastAPI service orchestrating execution, state management, and tool routing  
- **Agent Layer**: LangGraph workflow with pause/resume execution, tool planning, and PostgreSQL-backed persistence  

---

## ✨ Features

### 🛡️ Human-Gated Execution Layer
- Intercepts tool calls before execution  
- Supports pause/resume within agent workflow  
- Enables explicit approval or denial of actions  
- Forms a control boundary between LLM decisions and real-world effects  

---

### 🛠️ Dynamic MCP Registry (Hot Reload)
- Register MCP tools at runtime via JSON configs  
- Automatically reloads and recompiles the agent graph  
- Supports extensible capabilities without restarting the system 
- Strict schema validation for all tool definitions 

---

### 📚 Local RAG Integration
- PDF ingestion → chunking → embeddings (`nomic-embed-text`)  
- Stored in local ChromaDB  
- Retrieved via `query_knowledge_base` tool  

---

### 💾 Persistent Stateful Execution
- PostgreSQL-backed checkpointing using `AsyncPostgresSaver`  
- Multi-session support via `thread_id`  
- Enables resumable agent workflows  

---

### 🔄 Controlled Tool-Oriented Agent Design
- Clear separation:
  - Decision layer (LLM)
  - Execution layer (tools via MCP)
  - Control layer (HITL gating)  
- Designed for safe tool use in autonomous systems  

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
