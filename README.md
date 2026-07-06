# AI-Powered RAG Tool (OpenAI + Gemini)
## Research Context 
 This project is part of an ongoing research initiative under Professor Metoyer focused on evaluating Retrieval-Augmented Generation systems across different large language models, specifically between OpenAI API and Gemini API

## Tech Stack (WIP)
**Languages:** Python
**Backend Frameworks:** LangChain
**Database:** Pinecone (OpenAI pipeline), Chroma (Gemini pipeline)
### Environment Setup

Prerequisite: Git installed

1. Clone the repository:
```git clone https://github.com/pbae123/manual-rag-metoyerLab.git```

2. Create and activate a virtual environment:
```python3 -m venv venv```
```source .venv/bin/activate```

### Installation
1. Install required Python packages:
```pip install -r requirements.txt```

2. Create a `.env` file in the root directory to store environment variables:
```
OPENAI_API_KEY=
EMBED_MODEL=
CHAT_MODEL=
PINECONE_API_KEY=
GEMINI_API_KEY=
```

## Project Structure
- `src/` — OpenAI RAG pipeline (`rag.py`, `rag_utils.py`) and `benchmark.py`, which compares full-book vs. chapter-specific retrieval (latency, LLM-judge score, citation hit rate).
- `geminiRag/` — Gemini RAG pipeline (`rag_script.py`, `rag_utils.py`), used as the Gemini side of the OpenAI-vs-Gemini comparison. `extract_chapters.py` splits the source PDF into per-chapter PDFs for chapter-specific mode.

### To run 
- OpenAI RAG: `python src/rag.py`
- Gemini RAG: `python geminiRag/rag_script.py`
- Benchmark comparison: `python src/benchmark.py`
