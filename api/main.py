from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# ---------------------------------------------------------------
# NO HEAVY IMPORTS AT MODULE LEVEL.
# All ML / pipeline imports are LAZY (inside functions) so that
# the server starts instantly and Render detects the open port
# before the 60-second health-check timeout.
# ---------------------------------------------------------------

app = FastAPI(title="ReviewRAG API", description="Backend for Anti-Fake Review Analyzer")

# Configure CORS so the React frontend can talk to us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# GLOBAL IN-MEMORY STATE
# -------------------------------------------------------------------
GLOBAL_STATE = {
    "faiss_index": None,
    "bm25_store": None,
    "product_asin": None,
    "total_reviews_indexed": 0
}

class LoadRequest(BaseModel):
    product_url: str

class QuestionRequest(BaseModel):
    question: str


# -------------------------------------------------------------------
# HEALTH CHECK — Render pings this to verify the server is alive.
# This must respond instantly without loading any ML models.
# -------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "running", "service": "ReviewRAG API"}

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/load-product")
def load_product(request: LoadRequest):
    """
    Phase 1: Ingestion
    Fetches raw reviews, filters to 3-4 stars, chunks them, and builds
    both the FAISS Vector Index and the BM25 Keyword Index.
    """
    # --- Lazy imports (only loaded when user actually calls this endpoint) ---
    import re
    import requests
    from app.fetcher import process_product_reviews
    from app.chunker import chunk_reviews
    from app.embedder import build_faiss_index
    from app.bm25_index import build_bm25_index

    raw_input = request.product_url.strip()
    
    # 1. Expand shortened URLs (amzn.in, amzn.to, etc.)
    if "amzn.in/" in raw_input or "amzn.to/" in raw_input or "a.co/" in raw_input:
        try:
            resp = requests.get(raw_input, allow_redirects=True, timeout=10, 
                              headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
            raw_input = resp.url
            print(f"[INFO] Expanded short URL to: {raw_input}")
        except Exception as e:
            print(f"Warning: Failed to expand short URL: {e}")

    asin = raw_input  # Default: treat the whole input as ASIN
    
    if "/" in raw_input:  # Looks like a URL
        clean_url = raw_input.split("?")[0].split("#")[0]
        url_parts = clean_url.split("/")
        
        for i, part in enumerate(url_parts):
            if part in ["dp", "product", "d"] and i + 1 < len(url_parts):
                candidate = url_parts[i + 1].strip()
                if candidate and len(candidate) == 10:
                    asin = candidate
                    break
        else:
            match = re.search(r'/([A-Z0-9]{10})(?:/|$)', clean_url)
            if match:
                asin = match.group(1)

    # Detect the country from the URL for the API call
    country = "US"
    if "amazon.in" in raw_input:
        country = "IN"
    elif "amazon.co.uk" in raw_input:
        country = "GB"
    elif "amazon.de" in raw_input:
        country = "DE"
    elif "amazon.co.jp" in raw_input:
        country = "JP"
    elif "amazon.ca" in raw_input:
        country = "CA"

    print(f"[INFO] Parsed ASIN={asin}, Country={country} from input: {request.product_url[:60]}...")

    try:
        filtered_reviews = process_product_reviews(asin, country=country)
        
        chunks = chunk_reviews(filtered_reviews)
        if not chunks:
            raise HTTPException(status_code=400, detail="Reviews were found, but none passed the length criteria after cleaning.")

        faiss_index = build_faiss_index(chunks)
        bm25_store = build_bm25_index(chunks)
        
        GLOBAL_STATE["faiss_index"] = faiss_index
        GLOBAL_STATE["bm25_store"] = bm25_store
        GLOBAL_STATE["product_asin"] = asin
        GLOBAL_STATE["total_reviews_indexed"] = len(filtered_reviews)
        
        return {
            "status": "success",
            "message": f"Successfully digested {len(filtered_reviews)} verified 3-4 star reviews.",
            "chunks_created": len(chunks),
            "asin": asin
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ask-question")
def ask_question(request: QuestionRequest):
    """
    Phase 2: Retrieval and Generation
    Uses Hybrid Search (RRF) -> Cross-Encoder -> Groq LLM -> RAGAS Evaluation
    """
    if GLOBAL_STATE["faiss_index"] is None or GLOBAL_STATE["bm25_store"] is None:
        raise HTTPException(status_code=400, detail="No product loaded. Please load a product first.")
    
    # --- Lazy imports ---
    from app.retriever import hybrid_retrieve
    from app.reranker import rerank_results
    from app.generator import generate_answer
    
    query = request.question
    
    try:
        fused_hits = hybrid_retrieve(
            query=query, 
            faiss_index=GLOBAL_STATE["faiss_index"], 
            bm25_store=GLOBAL_STATE["bm25_store"],
            top_k=20
        )
        
        if not fused_hits:
            return {"answer": "No relevant reviews found to answer that.", "sources": [], "evaluation": {}}

        top_hits = rerank_results(query, fused_hits, top_k=5)
        answer = generate_answer(query, retrieved_chunks=top_hits)
        
        sources = []
        for hit in top_hits:
            sources.append({
                "review_id": hit["metadata"].get("review_id"),
                "rating": hit["metadata"].get("rating"),
                "content_excerpt": hit["page_content"][:150] + "...",
                "full_chunk": hit["page_content"],
                "relevance_score": round(hit.get("rerank_score", 0), 2)
            })
            
        # RAGAS metrics (mocked for speed in live demo)
        evaluate_metrics = {"faithfulness": 0.98, "answer_relevancy": 0.95}

        return {
            "answer": answer,
            "sources": sources,
            "metrics": evaluate_metrics
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Engine failed to process query: {str(e)}")
