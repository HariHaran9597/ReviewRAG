from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document

# Singleton instance for embeddings to avoid reloading
_embeddings_instance = None

def get_bge_embeddings():
    """
    Loads BAAI/bge-small-en-v1.5 model locally.
    Configured for CPU execution for zero-cost deployment.
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        model_name = "BAAI/bge-small-en-v1.5"
        encode_kwargs = {'normalize_embeddings': True} # Required for BGE to get similarity
        
        # Load local CPU model configuration
        _embeddings_instance = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs=encode_kwargs
        )
    return _embeddings_instance

def build_faiss_index(chunks: List[Dict[str, Any]]) -> FAISS:
    """
    Converts text chunks into dense embeddings, storing them directly into memory.
    Ensures zero external vector database costs.
    """
    # 1. Initialize embeddings model
    embeddings = get_bge_embeddings()
    
    # 2. Convert standard python dict chunks into Langchain Document objects
    documents = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk["page_content"],
            metadata=chunk["metadata"]
        )
        documents.append(doc)
        
    # 3. Create FAISS in-memory index
    faiss_index = FAISS.from_documents(documents, embeddings)
    
    # Returns an ephemeral index ready for retrieval
    return faiss_index
