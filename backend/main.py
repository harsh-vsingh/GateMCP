from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import backend.graph as graph
from .graph import setup_db, initialize_graph
from utils.history import get_chat_history, get_all_threads, delete_thread_history
from utils.pdf import process_pdf_to_vector_db
from utils.streaming import get_streaming_response, handle_denial
from utils.mcp_service import MCPService


app = FastAPI()
client = None
mcp_service = MCPService()


class ChatRequest(BaseModel):
    message: str
    thread_id: str

class ApprovalRequest(BaseModel):
    thread_id: str
    approved: bool


@app.on_event("startup")
async def startup_event():
    global client
    # DB Setup
    checkpointer, pool = await setup_db()
    await pool.open()
    await checkpointer.setup()
    
    # Dynamic Tool Loading from agent_workspace/mcp_servers/
    full_config = mcp_service.list_servers()
    client = await mcp_service.refresh_client(full_config)
    
    # Initial Graph Compilation
    tools = await client.get_tools()
    await initialize_graph(tools=tools)


@app.on_event("shutdown")
async def shutdown_event():
    if graph.pool:
        await graph.pool.close()
    if client:
        try: await client.close()
        except AttributeError: pass


# MCP Management Endpoints
async def _perform_hot_reload():
    """Internal helper to rebuild the client and graph."""
    global client
    full_config = mcp_service.list_servers()
    
    # Kills old subprocesses and starts new ones
    client = await mcp_service.refresh_client(full_config)
    
    # Re-binds tools to the LLM and re-compiles
    tools = await client.get_tools()
    await initialize_graph(tools=tools)
    return [t.name for t in tools]

@app.post("/mcp/save")
async def save_mcp(name: str, config: dict):
    """Stores config and automatically refreshes the agent."""
    try:
        # Validate the JSON schema and connectivity
        await mcp_service.validate_server(config)
        
        # Persist to agent_workspace/mcp_servers/
        mcp_service.save_server_config(name, config)
        
        # Trigger automatic refresh
        active_tools = await _perform_hot_reload()
        return {"status": "saved and refreshed", "active_tools": active_tools}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/mcp/list")
async def list_mcps():
    return mcp_service.list_servers()

@app.delete("/mcp/{name}")
async def delete_mcp(name: str):
    """Deletes config and automatically refreshes the agent."""
    mcp_service.delete_server_config(name)
    
    # Trigger automatic refresh 
    active_tools = await _perform_hot_reload()
    return {"status": "deleted and refreshed", "active_tools": active_tools}

@app.post("/mcp/refresh")
async def refresh_mcp():
    """Manual refresh endpoint (still available for safety)."""
    active_tools = await _perform_hot_reload()
    return {"status": "refreshed", "active_tools": active_tools}


# Chat & RAG Endpoints
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    chunks = await process_pdf_to_vector_db(file)
    return {"status": "success", "filename": file.filename, "chunks": chunks}

@app.post("/chat/stream")
async def chat_stream_endpoint(req: ChatRequest):
    if graph.chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    return StreamingResponse(
        get_streaming_response(req.message, req.thread_id), 
        media_type="text/event-stream" 
    )

@app.post("/chat/approve")
async def approve_tool(req: ApprovalRequest):
    if not req.approved:
        await handle_denial(req.thread_id)
    return StreamingResponse(
        get_streaming_response("", req.thread_id, is_resume=True), 
        media_type="text/event-stream"
    )

@app.get("/threads")
async def list_threads():
    return {"threads": await get_all_threads()}

@app.get("/history/{thread_id}")
async def history(thread_id: str):
    return {"history": await get_chat_history(thread_id)}

@app.delete("/history/{thread_id}")
async def delete_history(thread_id: str): 
    await delete_thread_history(thread_id)
    return {"status": "deleted"}


