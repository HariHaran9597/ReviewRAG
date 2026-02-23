# Use the official Python slim image for a smaller footprint
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for FAISS and compiling native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download HuggingFace models (BGE Embeddings and Reranker) during the build phase
# This prevents downloading them every time the container starts, making boot time instant.
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base', device='cpu')"
RUN python -c "from langchain_community.embeddings import HuggingFaceBgeEmbeddings; HuggingFaceBgeEmbeddings(model_name='BAAI/bge-small-en-v1.5', model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})"

# Copy the rest of the application code
COPY ./app ./app
COPY ./api ./api

# Set environment variables (these should be overridden in docker-compose or .env)
ENV RAPIDAPI_KEY=your_rapidapi_key
ENV GROQ_API_KEY=your_groq_api_key

# Expose the port the FastAPI app runs on
EXPOSE 8000

# Command to run the application using Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
