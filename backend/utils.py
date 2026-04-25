import json
import shutil
from pathlib import Path
from . import graph
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

SAFE_DIRECTORY = Path("/home/harsh/projects/chatbot/agent_workspace").resolve()
CHROMA_PATH = SAFE_DIRECTORY / "chroma_db"

def serialize_content(content):
    """Converts complex message content into a clean string for the UI."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join([item.get("text", "") if isinstance(item, dict) else str(item) for item in content])
    return str(content)

async def process_pdf_to_vector_db(file):
    """Handles temp file saving, chunking, and vector storage."""
    temp_path = Path(f"temp_{file.filename}")
    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        loader = PyPDFLoader(str(temp_path))
        pages = await loader.aload()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(pages)

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(CHROMA_PATH)
        )
        return len(chunks)
    finally:
        if temp_path.exists():
            temp_path.unlink()

async def get_chat_history(thread_id):
    """Fetches history using the async graph state."""
    if graph.chatbot is None:
        return []
    
    config = {'configurable': {'thread_id': str(thread_id)}}
    state = await graph.chatbot.aget_state(config=config)
    
    if not (state and state.values):
        return []
    
    messages = state.values.get('messages', [])
    ui_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            ui_messages.append({'role': 'user', 'content': msg.content})
        elif isinstance(msg, AIMessage) and msg.content:
            ui_messages.append({'role': 'assistant', 'content': msg.content})
        elif isinstance(msg, ToolMessage):
            ui_messages.append({'role': 'tool', 'content': f"Executed tool: {str(msg.name)}"})
    return ui_messages


async def get_all_threads():
    """Retrieves all unique thread IDs using the async pool."""
    async with graph.pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT DISTINCT thread_id FROM checkpoints")
            rows = await cur.fetchall()
            return [row[0] for row in rows]


async def delete_thread_history(thread_id):
    """Deletes all checkpoints for a thread using the async pool."""
    thread_id = str(thread_id)  
    async with graph.pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
            await cur.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))
            await cur.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
        await conn.commit()


async def get_streaming_response(message: str, thread_id: str, is_resume: bool = False):
    from .graph import chatbot
    config = {"configurable": {"thread_id": thread_id}}
    input_data = None if is_resume else {"messages": [("user", message)]}

    MAX_RETRIES = 2

    for attempt in range(MAX_RETRIES):
        try:
            async for event in chatbot.astream_events(input_data, config, version="v2"):
                kind = event["event"]

                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        yield json.dumps({
                            "type": "content",
                            "content": serialize_content(content)
                        }) + "\n"

                elif kind == "on_tool_start":
                    yield json.dumps({
                        "type": "tool_start",
                        "content": event["name"]
                    }) + "\n"

                elif kind == "on_tool_end":
                    yield json.dumps({
                        "type": "tool_end",
                        "content": event["name"]
                    }) + "\n"

            state = await chatbot.aget_state(config)
            if state.next and state.next[0] == "tools":
                last_msg = state.values["messages"][-1]
                tool_name = last_msg.tool_calls[0]["name"] if last_msg.tool_calls else "tool"
                yield json.dumps({
                    "type": "interrupt",
                    "content": tool_name
                }) + "\n"

            return 

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                continue

            yield json.dumps({
                "type": "error",
                "content": f"Streaming failed after {MAX_RETRIES} attempts: {str(e)}"
            }) + "\n"

async def handle_denial(thread_id: str):
    """Injects a cancellation message so the LLM knows the user denied the tool."""
    from .graph import chatbot
    config = {"configurable": {"thread_id": thread_id}}
    state = await chatbot.aget_state(config)
    
    tool_call_id = state.values["messages"][-1].tool_calls[0]["id"]
    cancellation_msg = ToolMessage(
        tool_call_id=tool_call_id,
        content="User denied permission to execute this tool."
    )
    await chatbot.aupdate_state(config, {"messages": [cancellation_msg]}, as_node="tools")