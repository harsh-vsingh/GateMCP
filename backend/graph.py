import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage
from psycopg_pool import AsyncConnectionPool


chatbot = None
pool = None
checkpointer = None

# Postgres checkpointer
async def setup_db():
    global pool, checkpointer
    DB_URI = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/langgraph_db")
    pool = AsyncConnectionPool(conninfo=DB_URI, open=False) 
    checkpointer = AsyncPostgresSaver(pool)
    return checkpointer, pool

# LLM setup
async def get_llm():
    MODEL_NAME = os.getenv("LLM_MODEL", "qwen3:8b") 
    MODEL_URL = os.getenv("MODEL_URL", "http://localhost:11434")
    llm = ChatOllama(model=MODEL_NAME, temperature=0, base_url=MODEL_URL)
    return llm


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

async def initialize_graph(tools=None):
    global chatbot
    llm = await get_llm()
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    async def chat_node(state: ChatState):
        response = await llm_with_tools.ainvoke(state['messages'])
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