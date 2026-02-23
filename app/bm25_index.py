from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

def build_bm25_index(chunks: List[Dict[str, Any]]) -> dict:
    """
    Builds a BM25 index over the chunks for exact keyword matching.
    Returns a dictionary containing the index and the original chunks for mapping.
    """
    if not chunks:
        return {"bm25": None, "chunks": []}
        
    # Tokenize the documents (simple whitespace splitting for BM25)
    tokenized_corpus = [chunk["page_content"].lower().split() for chunk in chunks]
    
    # Initialize the BM25 model
    bm25 = BM25Okapi(tokenized_corpus)
    
    return {
        "bm25": bm25,
        "chunks": chunks
    }

def query_bm25_index(bm25_store: dict, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """
    Queries the BM25 index and returns the top_k chunks.
    """
    bm25 = bm25_store.get("bm25")
    chunks = bm25_store.get("chunks", [])
    
    if not bm25 or not chunks:
        return []
        
    tokenized_query = query.lower().split()
    
    # Get the scores for the query against all documents
    doc_scores = bm25.get_scores(tokenized_query)
    
    # Get the indices of the top_k scores
    top_n_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
    
    results = []
    # Maximum BM25 score varies widely based on document length and query matching.
    # We add a synthetic "score" so we can map it later in our fusion algo.
    max_score = float(max(doc_scores)) if len(doc_scores) > 0 else 1.0
    if max_score == 0:
        max_score = 1.0
        
    for rank, idx in enumerate(top_n_indices):
        if doc_scores[idx] > 0: # Only return actual matches
            chunk = chunks[idx].copy()
            # Normalize score relative to the max score found
            chunk["bm25_score"] = doc_scores[idx] / max_score
            results.append(chunk)
            
    return results
