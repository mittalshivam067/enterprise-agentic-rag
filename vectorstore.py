import os
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables from .env
load_dotenv()

PERSIST_DIRECTORY = "./chroma_db"

def get_embedding_function() -> GoogleGenerativeAIEmbeddings:
    """Initialize the Gemini embedding model."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=api_key)

def build_vector_store(chunks: List[Document], persist_dir: str = PERSIST_DIRECTORY) -> Chroma:
    """
    Takes documents chunks, computes embeddings, and stores them in a Chroma vector store.
    Persists to disk automatically.
    """
    embeddings = get_embedding_function()
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    return vector_db

def load_existing_vector_store(persist_dir: str = PERSIST_DIRECTORY) -> Chroma:
    """Loads an existing ChromaDB instance from the local persist directory."""
    embeddings = get_embedding_function()
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

def similarity_search_with_filter(
        vector_db: Chroma,
        query: str,
        top_k: int = 4,
        score_threshold: float = 0.55
) -> List[Tuple[Document, float]]:
    """
    Searches for the most relevant document chunks based on semantic similarity.
    Filters out chunks with scores below the relevance threshold to reduce noise.
    """
    # Chroma returns cosine similarity scores between 0 and 1
    results = vector_db.similarity_search_with_relevance_scores(query, k=top_k)

    # Filter by relevance threshold
    filtered_results = [(doc, score) for doc, score in results if score >= score_threshold]

    # If no chunk passes the threshold, fall back to the single top match (if available)
    if not filtered_results and results:
        filtered_results = [results[0]]

    return filtered_results

if __name__ == "__main__":
    # Test vectorstore indexing and retrieval
    print("Testing Vector Store setup...")
    sample_docs = [
        Document(
            page_content="Project Alpha: Enterprise quarterly revenue grew by 18% in Q3.",
            metadata={"source": "report_q3.txt", "type": "txt"}
        ),
        Document(
            page_content="Server deployment guidelines: Ensure all Docker containers run on port 8080.",
            metadata={"source": "devops_guide.txt", "type": "txt"}
        )
    ]

    # Index test documents
    vdb = build_vector_store(sample_docs)
    print("Vector Store successfully created.")
    
    # Run test search
    test_query = "What was the enterprise revenue growth in Q3?"
    matches = similarity_search_with_filter(vdb, test_query)
    
    print(f"\nQuery: '{test_query}'")
    for doc, score in matches:
        print(f"-> Score: {score:.4f} | Source: {doc.metadata['source']}")
        print(f"   Content: {doc.page_content}")