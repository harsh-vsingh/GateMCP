import shutil
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

SAFE_DIRECTORY = Path(__file__).parent.parent.parent.resolve()
CHROMA_PATH = SAFE_DIRECTORY / "agent_workspace" /"chroma_db"


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
        for c in chunks:
            c.metadata["source"] = file.filename

        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        vector_db = Chroma(
            persist_directory=str(CHROMA_PATH),
            embedding_function=embeddings
        )

        existing = vector_db.get(where={"source": file.filename}, limit=1)

        if existing and existing.get("ids"):
            vector_db.delete(where={"source": file.filename})
        
        vector_db.add_documents(chunks)
        vector_db.persist()
        return len(chunks)
    finally:
        if temp_path.exists():
            temp_path.unlink()

async def delete_pdf_from_vector_db(filename: str):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embeddings
    )

    vector_db.delete(where={"source": filename})
    vector_db.persist()

async def list_indexed_pdfs():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embeddings
    )

    data = vector_db.get()

    if not data or not data.get("metadatas"):
        return []

    sources = set()
    for m in data["metadatas"]:
        if m and "source" in m:
            sources.add(m["source"])

    return list(sources)