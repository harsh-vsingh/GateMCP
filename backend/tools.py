from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os
from fastmcp import FastMCP
from tavily import TavilyClient
from dotenv import load_dotenv
import aiofiles
from pathlib import Path

load_dotenv()

# Tools
mcp = FastMCP("tls")

_tavily = None
def get_client():
    global _tavily
    if _tavily is None:
        _tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return _tavily

@mcp.tool
def search_internet(query: str) -> str:
    """
    Search the internet using Tavily.
    """
    try:
        client = get_client()

        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=3,
            include_answer=True
        )

        results = response.get("results", [])
        answer = response.get("answer", "")

        if not results:
            return "No results found."

        formatted = []

        if answer:
            formatted.append(f"Answer:\n{answer}")

        for r in results:
            formatted.append(
                f"Title: {r.get('title')}\n"
                f"Snippet: {r.get('content')}\n"
                f"URL: {r.get('url')}"
            )

        return "\n\n---\n\n".join(formatted)

    except Exception as e:
        return f"Search error: {type(e).__name__} - {str(e)}"




SAFE_DIRECTORY = Path("/home/harsh/projects/chatbot/agent_workspace").resolve()
SAFE_DIRECTORY.mkdir(parents=True, exist_ok=True)

def is_safe_path(path: str) -> bool:
    """Check if the resolved path is within the SAFE_DIRECTORY."""
    try:
        resolved_path = (SAFE_DIRECTORY / path).resolve()
        return SAFE_DIRECTORY in resolved_path.parents or resolved_path == SAFE_DIRECTORY
    except Exception:
        return False
    
@mcp.tool
async def list_files() -> str:
    """Lists all files in the allowed workspace."""
    files = os.listdir(SAFE_DIRECTORY)
    return "\n".join(files) if files else "Workspace is empty."

@mcp.tool
async def write_to_file(filename: str, content: str) -> str:
    """Writes content to a specific file in the workspace."""
    if not is_safe_path(filename):
        return "Error: Access denied. You can only write to the designated workspace."
    
    file_path = SAFE_DIRECTORY / filename
    async with aiofiles.open(file_path, mode='w') as f:
        await f.write(content)
    return f"Successfully wrote to {filename}"

@mcp.tool
async def read_from_file(filename: str) -> str:
    """Reads content from a specific file in the workspace."""
    if not is_safe_path(filename):
        return "Error: Access denied. You can only read from the designated workspace."
    
    file_path = SAFE_DIRECTORY / filename
    if not file_path.exists():
        return f"Error: {filename} does not exist."
        
    async with aiofiles.open(file_path, mode='r') as f:
        content = await f.read()
    return content

@mcp.tool
async def query_knowledge_base(query: str) -> str:
    """
    Search your uploaded local documents for relevant information. 
    Use this when the user asks about PDFs they have uploaded.
    """
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector_db = Chroma(
            persist_directory="/home/harsh/projects/chatbot/agent_workspace/chroma_db",
            embedding_function=embeddings
        )
        
        docs = vector_db.similarity_search(query, k=3)
        
        if not docs:
            return "No relevant information found in the uploaded documents."
            
        return "\n\n---\n\n".join([d.page_content for d in docs])
    except Exception as e:
        return f"Knowledge base error: {str(e)}"

if __name__ == "__main__":
    mcp.run()