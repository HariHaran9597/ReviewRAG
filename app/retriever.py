from typing import List, Dict, Any
from collections import defaultdict
from langchain_community.vectorstores import FAISS

def hybrid_retrieve(
    query: str,
    faiss_index: FAISS,
    bm25_store: dict,
    top_k: int = 20,
    rrf_k: int = 60
) -> List[Dict[str, Any]]:
    """
    Performs Hybrid Search by querying FAISS (Dense) and BM25 (Sparse) independently,
    then combines the results using Reciprocal Rank Fusion (RRF).
    """

    # 1. FAISS Vector Search
    vector_results = faiss_index.similarity_search_with_score(query, k=top_k)
    
    vector_hits = []
    # Note: similarity_search_with_score in FAISS returns (Document, L2_distance).
    # Lower distance is better.
    for doc, dist in vector_results:
        hit = {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
            "vector_dist": dist
        }
        vector_hits.append(hit)

    # 2. BM25 Keyword Search
    from app.bm25_index import query_bm25_index
    bm25_hits = query_bm25_index(bm25_store, query, top_k=top_k)

    # 3. Reciprocal Rank Fusion (RRF)
    # RRF Score = 1 / (rank + k)
    # We use a unique identified for each chunk based on review_id and chunk_index
    
    chunk_map = {}
    rrf_scores = defaultdict(float)

    # Add Vector hits to RRF
    for rank, hit in enumerate(vector_hits):
        # Create a unique ID to identify duplicates
        rev_id = hit["metadata"].get("review_id", "")
        c_idx = hit["metadata"].get("chunk_index", 0)
        uid = f"{rev_id}_{c_idx}"
        
        chunk_map[uid] = hit # store the payload
        rrf_scores[uid] += 1.0 / (rank + 1 + rrf_k)

    # Add BM25 hits to RRF
    for rank, hit in enumerate(bm25_hits):
        rev_id = hit["metadata"].get("review_id", "")
        c_idx = hit["metadata"].get("chunk_index", 0)
        uid = f"{rev_id}_{c_idx}"
        
        if uid not in chunk_map:
            chunk_map[uid] = hit
            
        rrf_scores[uid] += 1.0 / (rank + 1 + rrf_k)

    # Sort by combined RRF score
    sorted_uids = sorted(rrf_scores.keys(), key=lambda uid: rrf_scores[uid], reverse=True)
    
    # Return the highly ranked chunks
    fused_results = []
    for uid in sorted_uids[:top_k]:
        hit = chunk_map[uid]
        hit["rrf_score"] = rrf_scores[uid]
        fused_results.append(hit)
        
    return fused_results
