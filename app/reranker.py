from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

_reranker_instance = None

def get_bge_reranker():
    """
    Loads the BGE Cross-Encoder locally.
    Configured for CPU execution for zero-cost operation.
    """
    global _reranker_instance
    if _reranker_instance is None:
        model_name = "BAAI/bge-reranker-base"
        _reranker_instance = CrossEncoder(model_name, device="cpu")
    return _reranker_instance

def rerank_results(query: str, hits: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Takes the initial top_k results from hybrid search and uses a cross-encoder
    to re-score and return the final top_k results.
    """
    if not hits:
        return []
        
    reranker = get_bge_reranker()
    
    # Create query-document pairs for the format required by the CrossEncoder
    pairs = []
    for hit in hits:
        # Cross-encoder expects [(query, text_1), (query, text_2), ...]
        pairs.append([query, hit["page_content"]])
        
    # Get relevancy scores directly from the model
    scores = reranker.predict(pairs)
    
    # Embed the scores into the payload dictionary
    for i, score in enumerate(scores):
        hits[i]["rerank_score"] = float(score)

    # Sort chunks in descending order based on the new Reranker score
    sorted_hits = sorted(hits, key=lambda x: x["rerank_score"], reverse=True)
    
    # Return the highly accurate top_k to pass to our LLM for final generation
    return sorted_hits[:top_k]
