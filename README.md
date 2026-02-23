# ReviewRAG: Anti-Fake Review Analyzer

<img width="2445" height="1257" alt="image" src="https://github.com/user-attachments/assets/ec71a264-b41c-4d57-a910-b55f15ce437e" />


ReviewRAG is an intelligent, full-stack application designed to cut through fake, promotional 5-star reviews and unhelpful 1-star shipping complaints on e-commerce platforms like Amazon and Flipkart. 

By aggressively filtering down to ONLY verified 3 and 4-star purchases, and utilizing an advanced **In-Memory Hybrid Search Pipeline (FAISS + BM25 + Reciprocal Rank Fusion + BGE Reranker)**, it extracts the absolute ground-truth about a product. The final answer is synthesized using the lightning-fast `moonshotai/kimi-k2-instruct-0905` model via Groq, with built-in RAGAS metric evaluation.

## 🌟 Key Features

1. **Strict Data Filtering**: Ignores noisy 5-star and 1-star reviews. Only processes verified 3-star and 4-star reviews longer than 20 words for maximum insight.
2. **Hybrid Retrieval (RRF)**: Combines FAISS Vector Search with BM25 Keyword Search using Reciprocal Rank Fusion to never miss exact product features.
3. **Cross-Encoder Reranking**: Rescores the top retrieved results using a `BAAI/bge-reranker-base` cross-encoder for peak precision.
4. **Ephemeral Memory**: Uses a zero-cost, temporary In-Memory FAISS index for each user session instead of an expensive persistent database.
5. **Quality Assured**: Answers are evaluated post-generation for *Faithfulness* and *Answer Relevancy* using RAGAS concepts.
6. **Smart Fallbacks**: Robustly handles shortened URLs, regional Amazon links (like `.in`, `.co.uk`), and gracefully handles API limitations.


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
