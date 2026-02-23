# ReviewRAG: Anti-Fake Review Analyzer

![Project Header Banner] <!-- Add a screenshot of the Glassmorphism UI here -->

**A Production-Grade Generative AI Engineering Portfolio Project.**

ReviewRAG is an intelligent, full-stack application designed to cut through fake, promotional 5-star reviews and unhelpful 1-star shipping complaints on e-commerce platforms like Amazon and Flipkart. 

By aggressively filtering down to ONLY verified 3 and 4-star purchases, and utilizing an advanced **In-Memory Hybrid Search Pipeline (FAISS + BM25 + Reciprocal Rank Fusion + BGE Reranker)**, it extracts the absolute ground-truth about a product. The final answer is synthesized using the lightning-fast `moonshotai/kimi-k2-instruct-0905` model via Groq, with built-in RAGAS metric evaluation.

## 🌟 Why This Project Stands Out (Architecture Decisions)

When evaluating AI Candidates, recruiters and senior engineers look for three things: Data Quality understanding, Retrieval complexity, and Cost-awareness. This application hits all three:

1. **Strict Data Filtering over "More Data"**: Standard tutorials feed all text to an LLM. ReviewRAG intentionally discards 5-star (often paid) and 1-star (often user-error) reviews. It only embeds verified 3-star and 4-star reviews longer than 20 words, proving an elite understanding that **Signal > Noise**.
2. **Hybrid Retrieval with Reciprocal Rank Fusion (RRF)**: Instead of basic dense vector search (which often misses exact part numbers or specific feature requests), this project simultaneously queries a FAISS Vector Index and a sparse BM25 Keyword index, merging them via the RRF mathematical algorithm.
3. **Cross-Encoder Reranking**: The top 20 fused results are rescored using a `BAAI/bge-reranker-base` cross-encoder for peak precision before hitting the LLM.
4. **$0.00 Operating Cost (Ephemeral Memory Strategy)**: Because every product's reviews are isolated datasets, paying for a persistent database (like Pinecone) is architecturally incorrect. ReviewRAG spins up a temporary In-Memory FAISS index for each user session.
5. **RAGAS Evaluated**: Answers are evaluated post-generation for *Faithfulness* and *Answer Relevancy*, proving the system does not hallucinate. OpenAI dependencies were bypassed to run this evaluation entirely free via Groq.

## ⚙️ Tech Stack

- **Backend / API**: FastAPI, Python 3.10
- **Frontend**: React (Vite), Custom CSS (Glassmorphism styling), Lucide-React
- **Data Source**: RapidAPI (Amazon Product Reviews API)
- **Vector DB**: FAISS (In-Memory CPU)
- **Sparse DB**: `rank_bm25`
- **Embeddings & Reranker**: `BAAI/bge-small-en-v1.5` & `BAAI/bge-reranker-base` (Running locally on CPU)
- **LLM Engine**: Groq API (`moonshotai/kimi-k2-instruct-0905`)
- **Evaluation**: RAGAS Framework 

## 🚀 Quick Start (Run Locally)

This application is fully containerized. You only need Docker installed on your machine.

### 1. Set your API Keys
Open the `.env` file in the root directory and add your API keys:
```env
RAPIDAPI_KEY=your_rapidapi_key
GROQ_API_KEY=your_groq_api_key
```
*(Get the RapidAPI key from any reliable Amazon Scraper on rapidapi.com, and Groq keys are free at console.groq.com)*

### 2. Boot the Application
Run the following command in the root folder:
```bash
docker-compose up --build
```
*Note: The first build will take a bit longer as it pre-downloads the HuggingFace BGE models into the image so they boot instantly on future runs.*

### 3. Access the UI
Open your browser and navigate to:
**http://localhost:5173**

## 📂 Project Structure
```text
review_rag/
├── app/
│   ├── fetcher.py          # RapidAPI integration with robust error boundaries
│   ├── chunker.py          # HTML/Emoji stripping and Recursive Text Splitting
│   ├── embedder.py         # FAISS Index compilation
│   ├── bm25_index.py       # Sparse keyword index logic
│   ├── retriever.py        # Hybrid Search Engine + RRF Fusion
│   ├── reranker.py         # Cross-Encoder logic
│   ├── generator.py        # Groq LLM (moonshotai/kimi-k2-instruct-0905)
│   └── evaluator.py        # Custom RAGAS wrapper bypassing OpenAI
├── api/
│   └── main.py             # FastAPI REST endpoints
├── frontend/               # Vite + React (Glassmorphism Dashboard)
│   └── src/
├── docker-compose.yml
├── Dockerfile              # Configured to pre-download local weights
└── requirements.txt
```

## 📝 Demo Flow
1. Paste `B08XJG8KVG` (or any Amazon ASIN/URL) into the UI.
2. The Dashboard will show the "Product Insight Engine" spin up, indicating System Readiness and how many "Knowledge Chunks" it generated from the verified 3/4 star reviews.
3. Ask a hard query: *"Are there complaints about the hinge breaking?"*
4. The system will hit the Hybrid Search, Rerank, and stream back the absolute truth alongside **Citation Badges** showing the exact star-rating and text snippet it used to verify its claim.
