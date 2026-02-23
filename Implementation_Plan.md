================================================================
REVIEW-RAG: ANTI-FAKE REVIEW ANALYZER
Updated End-To-End Implementation Plan (React + Groq + RAGAS)
Production-Grade RAG Project — Applied AI Engineer Resume
================================================================

----------------------------------------------------------------
0. WHAT ARE WE BUILDING
----------------------------------------------------------------

THE PROBLEM:
You open an Amazon or Flipkart product page. 8,000 reviews. You want to know one thing: "Does this blender actually crush ice?" You scroll for 20 minutes. You find nothing useful. 5-star reviews are paid promotions. 1-star reviews are shipping complaints. Real signal is buried in 3-star and 4-star verified reviews.

THE SOLUTION:
A stunning React/FastAPI web application where you paste any Amazon or Flipkart product URL. The backend fetches ONLY verified 3-4 star reviews using RapidAPI, gracefully handling anti-bot blocks. It chunks and embeds the text using a hybrid search engine (BGE Vector + BM25 keyword fusion). You ask specific questions in natural language and get direct, cited answers scored by RAGAS (configured to run 100% free via Groq/moonshotai/kimi-k2-instruct-0905)—ensuring no hallucination and absolute truth.

WHY THIS WINS IN INTERVIEWS:
- The 3-4 star filter proves deep data-quality understanding (Senior trait).
- Hybrid search + Reranking shows mastery of advanced retrieval over basic dense vectors.
- Overcoming RapidAPI scraping limits gracefully proves real-world engineering resilience.
- RAGAS evaluation (hacked to run free) shows rigorous metric-driven development.
- The beautiful React Frontend ensures the live demo WOWs recruiters instantly.

----------------------------------------------------------------
1. SYSTEM ARCHITECTURE
----------------------------------------------------------------

TWO PHASES:

PHASE 1 — INGESTION (when user pastes URL):
Step 1:  Fetch Reviews      — RapidAPI call, gracefully catching timeouts/blocks
Step 2:  Filter Reviews     — Keep ONLY verified 3-star and 4-star
Step 3:  Clean Text         — Remove HTML, emojis, duplicates, <20 word reviews
Step 4:  Chunk Reviews      — Split into 150-200 word chunks with metadata
Step 5:  Embed Chunks       — BGE-small model, run locally on CPU
Step 6:  Build BM25 Index   — Keyword index on the same chunks
Step 7:  Store in FAISS     — In-memory index for this session only (lowest latency)

PHASE 2 — QUERYING (when user asks a question):
Step 8:  Receive Question   — FastAPI endpoint gets user query
Step 9:  Hybrid Retrieve    — Vector search AND BM25 simultaneously, top 20 each
Step 10: RRF Fusion         — Merge both lists using Reciprocal Rank Fusion
Step 11: Rerank             — BGE cross-encoder rescores top 20, returns top 5
Step 12: Generate Answer    — Groq LLM synthesizes answer from top 5 chunks
Step 13: RAGAS Evaluate     — Custom wrapper to score Faithfulness & Relevancy via Groq
Step 14: Return Response    — FastAPI returns Answer + Sources + RAGAS Scores to React

PROJECT FOLDER STRUCTURE:
review_rag/
├── app/
│   ├── __init__.py
│   ├── fetcher.py          # RapidAPI integration with robust fallback/error handling
│   ├── chunker.py          # Text cleaning and metadata chunking
│   ├── embedder.py         # BGE embeddings + FAISS index logic
│   ├── bm25_index.py       # BM25 keyword index logic
│   ├── retriever.py        # Hybrid search + RRF fusion
│   ├── reranker.py         # BGE cross-encoder reranker
│   ├── generator.py        # Groq LLM configuration for answers
│   └── evaluator.py        # Custom RAGAS wrapper (configured for moonshotai/kimi-k2-instruct-0905, bypassing OpenAI)
├── api/
│   └── main.py             # FastAPI routes (handles CORS for React frontend)
├── frontend/               # VITE + REACT
│   └── src/                # Glassmorphism UI, chat components, source citation badges
├── Dockerfile              # Containerizes FastAPI backend
├── docker-compose.yml      # Orchestrates Backend + Frontend deployment
├── requirements.txt
└── README.md

----------------------------------------------------------------
2. FINAL TECH STACK (All Free, Zero Cost)
----------------------------------------------------------------

Backend Core        : FastAPI + Uvicorn
Frontend            : React (Vite) + Custom CSS (Glassmorphism)
Review Fetching     : RapidAPI Amazon Reviews API (Free tier, with `try/except` fallbacks)
Text Embedding      : BAAI/bge-small-en-v1.5 (Local CPU, fast)
Vector Search       : FAISS In-Memory (Zero storage cost, per-session ephemeral)
Keyword Search      : rank_bm25 (Library)
Hybrid Fusion       : Custom Reciprocal Rank Fusion (Python)
Reranking           : BAAI/bge-reranker-base (Local CPU)
LLM (Generator)     : Groq API (moonshotai/kimi-k2-instruct-0905, fast & free)
LLM (Evaluator)     : Groq API wired into RAGAS (to bypass OpenAI costs)
Evaluation          : RAGAS (Industry standard framework)
Deployment          : Render (Free tier) or Hugging Face Spaces

----------------------------------------------------------------
3. DAY-BY-DAY BUILD PLAN (4 DAYS TO LIVE)
----------------------------------------------------------------

DAY 1 — Foundation: Ingestion, Filtering, and FAISS
Goal: Robust backend ingestion that survives API blocks.
- Set up python virtual environment, initialize Git.
- Implement `app/fetcher.py`: Connect to RapidAPI. Write strong error boundaries so if Amazon blocks the IP, it returns a 400 error cleanly to the frontend instead of crashing.
- Implement strictly: Keep only "Verified" + "3 or 4 Stars" + ">20 words".
- Build `app/chunker.py` and `app/embedder.py` to convert cleaned reviews into an in-memory FAISS index.

DAY 2 — The "Senior" Upgrades: Hybrid Search & RAGAS Evaluation
Goal: Elite retrieval precision and zero-cost evaluation metric setup.
- Build `app/bm25_index.py` for keyword matching.
- Implement `app/retriever.py` to query FAISS + BM25 simultaneously and merge results using the Reciprocal Rank Fusion math formula.
- Implement `app/reranker.py` with Cross-Encoder.
- Build `app/evaluator.py`: THIS IS CRITICAL. Write the custom Langchain wrapper to force the `ragas` library to use `ChatGroq(moonshotai/kimi-k2-instruct-0905)` and our local CPU embeddings instead of crashing asking for an OpenAI key.

DAY 3 — The Brains: FastAPI & The React Frontend
Goal: Connect the robust backend to a stunning UI.
- Build `api/main.py` creating the specific endpoints: `/api/load-product` and `/api/ask-question`.
- Scaffold the Vite React App (`/frontend`).
- Build the Glassmorphism Dashboard: A clean URL input screen that transitions into a split-pane view (Product details on left, Chat window on right).
- Implement source citation badges in the chat bubbles so users can read the original reviews driving the answer.

DAY 4 — Dockerization, Deployment, & Documentation
Goal: A shareable live URL for recruiters.
- Write the `Dockerfile` to containerize the FastAPI backend.
- Configure `docker-compose.yml` to serve the React frontend alongside it.
- Write the elite README.md detailing the architecture and the "Why I didn't use Pinecone/OpenAI" interview logic.
- Deploy to Render.

----------------------------------------------------------------
4. THE INTERVIEW "ELEVATOR PITCH" (Memorize)
----------------------------------------------------------------

"I wanted to build a RAG system that solved a daily problem: navigating fake 5-star reviews on Amazon. To get high-quality truth, my system filters out everything except verified 3 and 4-star reviews. For retrieval, I bypassed standard vector search and engineered a Hybrid Pipeline—combining FAISS with BM25 keyword search, fused via Reciprocal Rank Fusion, and rescored by a BGE Cross-Encoder. Because every product is a completely isolated dataset, I architected this to run entirely In-Memory, sidestepping the latency and cost of persistent databases like Pinecone. Finally, I hacked the RAGAS framework to run evaluated Faithfulness and Answer Relevancy metrics entirely freely via Groq. It’s fully containerized via FastAPI and Docker with a React frontend."
