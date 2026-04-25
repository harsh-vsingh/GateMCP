import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, SystemMessage
from psycopg_pool import AsyncConnectionPool


chatbot = None
pool = None
checkpointer = None

# Postgres checkpointer
async def setup_db():
    global pool, checkpointer
    DB_URI = os.getenv("DATABASE_URL")
    pool = AsyncConnectionPool(conninfo=DB_URI, open=False) 
    checkpointer = AsyncPostgresSaver(pool)
    return checkpointer, pool

# LLM setup
async def get_llm():
    MODEL_NAME = os.getenv("LLM_MODEL") 
    MODEL_URL = os.getenv("MODEL_URL")
    llm = ChatOllama(model=MODEL_NAME, temperature=0, base_url=MODEL_URL)
    return llm


SYSTEM_PROMPT = """
You are an AI agent with access to a dynamic set of tools.

Your primary goal is to solve tasks by effectively using available tools.

General rules:

1. Tool Usage
- Use tools whenever they help produce a better, more accurate answer.
- Do not rely only on your internal knowledge if a tool is more appropriate.
- Choose the most relevant tool based on its description.

2. Dynamic Tools
- Tools may vary at runtime. Always reason about what each tool does before using it.
- Do not assume fixed tool names except when explicitly relevant.

3. Human Approval (Critical)
- Tool execution may require user approval.
- When a tool is appropriate, call it confidently and let the system handle approval.
- Do not avoid tools due to approval requirements.

4. Knowledge Base (RAG)
- If the query relates to uploaded files, documents, or user-provided data, prefer the knowledge base tool.
- If no relevant information is found, explicitly say so. Do not fabricate.

5. Internet / External Data
- Use internet-based tools when the query requires up-to-date or external information.

6. Avoid Hallucination
- If unsure and no tool can help, say you don’t know.
- Never invent facts.

7. Efficiency
- Do not call tools unnecessarily.
- Do not call multiple tools unless needed.

8. Response Style
- Be concise, factual, and direct.
"""

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

async def initialize_graph(tools=None):
    global chatbot
    llm = await get_llm()
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    async def chat_node(state: ChatState):
        messages = state['messages']
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        response = await llm_with_tools.ainvoke(messages)
        return {'messages': [response]}
 
    workflow = StateGraph(ChatState)
    workflow.add_node('chat_node', chat_node)
    
    if tools:
        workflow.add_node('tools', ToolNode(tools))
        workflow.add_edge(START, 'chat_node')
        workflow.add_conditional_edges('chat_node', tools_condition)
        workflow.add_edge('tools', 'chat_node')
    else:
        workflow.add_edge(START, 'chat_node')

    chatbot = workflow.compile(checkpointer=checkpointer, interrupt_before=['tools'] if tools else None)
    return chatbot