from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()
from pathlib import Path
from fastmcp import FastMCP
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

SAFE_DIRECTORY = Path(__file__).parent.parent.resolve()
CHROMA_PATH = SAFE_DIRECTORY / "agent_workspace" /"chroma_db"

# Built in internet and RAG tools

# Internet tool using tavily
_tavily = None
def get_client():
    global _tavily
    if _tavily is None:
        _tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return _tavily

mcp = FastMCP("tls")

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


# RAG tool
@mcp.tool
async def query_knowledge_base(query: str) -> str:
    """
    Search your uploaded local documents for relevant information. 
    Use this when the user asks about PDFs they have uploaded.
    """
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector_db = Chroma(
            persist_directory=str(CHROMA_PATH),
            embedding_function=embeddings
        )
        
        results = vector_db.similarity_search_with_score(query, k=5)

        threshold = 0.5
        docs = [doc for doc, score in results if score < threshold]
        docs = docs[:3]
        
        if not docs:
            return "No relevant information found in the uploaded documents."

        return "\n\n---\n\n".join([d.page_content for d in docs])
    except Exception as e:
        return f"Knowledge base error: {str(e)}"

if __name__ == "__main__":
    mcp.run()