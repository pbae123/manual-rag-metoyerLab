# AI-Powered (OpenAI API) Rag Tool
## Research Context 
 This project is part of an ongoing research initiative under Professor Metoyer focused on evaluating Retrieval-Augmented Generation systems across different large language models, specifically between OpenAI API and Gemini API

## Tech Stack (WIP)
**Languages:** Python
**Backend Frameworks:** LangChain
**Database:** Pinecone
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
```

### To run 
python src/rag.py
