# RAG Cognitive Caching Service

A high-performance, asynchronous Python cognitive caching service designed to sit between a web frontend and LLM APIs (e.g. Gemini). It intercepts incoming prompts to perform **Vector Similarity Search (VSS)** against cached queries, bypassing LLM inference entirely on semantic hits to serve cached documents in under 15ms.

## Features

- **Semantic Vector Cache**: Performs dense similarity search using local sentence embeddings (cosine similarity > 0.90).
- **FastAPI Core Router**: Clean asynchronous routing gateway for high-throughput caching.
- **Offline Embeddings Engine**: Uses local `SentenceTransformers` (`all-MiniLM-L6-v2`) to eliminate translation billing costs and network latency.
- **Vector Store Database**: Hosts vector index collections using an in-memory/persisted Qdrant collection setup.

## Project Structure

```
├── main.py                 # FastAPI application and route definitions
├── services/
│   ├── embedder.py         # Dense vector generation (all-MiniLM-L6-v2)
│   └── vector_store.py     # Qdrant client connection and query index rules
├── Makefile                # Automation commands (setup, start, clean)
├── requirements.txt        # Python dependency manifest
└── .gitignore              # Python git exclusions
```

## Setup & Running

1. **Setup Environment**:
   Initialize the virtual environment and install all packages:
   ```bash
   make setup
   ```

2. **Start the Service**:
   Run the FastAPI server on port 8000:
   ```bash
   make start
   ```

3. **Clean Cache files**:
   ```bash
   make clean
   ```
