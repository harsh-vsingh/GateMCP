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




