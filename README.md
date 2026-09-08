# AI Agent with RAG + Live API Tools

A production-style agentic AI system built with **LangGraph, LangChain, Google Gemini, and FastAPI**. It answers user questions by combining three capabilities:

1. **Knowledge-base search (RAG)** — retrieves answers from a local document store (ChromaDB)
2. **Live API calls** — fetches real-time weather from the Open-Meteo API
3. **Plain reasoning** — answers general questions directly, no tools needed

The agent decides on its own when to search the knowledge base, when to call the API, or when to just answer. Everything is exposed through a clean REST API.

## Architecture

```
User question
      │
      ▼
POST /api/v1/chat  (FastAPI)
      │
      ▼
┌─────────────────────────────┐
│  LangGraph Agent (ReAct)    │
│  ┌───────────┐              │
│  │ chatbot   │  LLM decides │
│  │ (Gemini)  │  tool or not │
│  └─────┬─────┘              │
│        │ tool call          │
│        ▼                    │
│  ┌───────────┐              │
│  │ tools     │ RAG / weather│
│  └─────┬─────┘              │
│        └───────► back to chatbot until answer ready
└─────────────────────────────┘
      │
      ▼
{ "status", "response", "tools_used" }
```

- `ingest.py` — splits `data/*.txt` into chunks and stores embeddings in ChromaDB
- `rag_tool.py` — retrieves the top chunks for a question (RAG retrieval)
- `api_tool.py` — live weather via Open-Meteo
- `agent.py` — the LangGraph ReAct agent that ties it together
- `main.py` — FastAPI server

## Prerequisites

- Python 3.14+
- A free Gemini API key from [Google AI Studio](https://aistudio.google.com)
- Docker (optional, for containerized deployment)

## Quick Start (local)

```bash
# 1. Create venv and activate
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
cp .env.example .env
# ... open .env and paste your GEMINI_API_KEY

# 4. Put your knowledge documents in data/ (txt files)

# 5. Build the knowledge base (first-time indexing)
python ingest.py

# 6. Start the server
uvicorn main:app --reload
```

The API is now live at `http://localhost:8000` — interactive docs at `http://localhost:8000/docs`.

## Quick Start (Docker)

```bash
docker compose up --build
```

The entrypoint indexes your documents automatically on first run and skips it afterwards.

## API Reference

### `POST /api/v1/chat`

Ask the agent anything.

Request:
```json
{ "message": "What is the policy on lasers?" }
```

Response:
```json
{
  "status": "success",
  "response": "According to the safety guidelines...",
  "tools_used": ["knowledge_base"]
}
```

`tools_used` reports which tools the agent executed — `knowledge_base`, `weather`, or `[]` (answered directly).

### `GET /health`

Liveness check → `{"status": "ok"}`

## Example Queries

| Question | What happens |
|----------|--------------|
| "What is the policy on lasers?" | searches the knowledge base |
| "What is the weather in Tokyo?" | calls the live weather API |
| "What is 12 times 8?" | answers directly (no tools) |

## Notes

- Free-tier Gemini quotas are small (20 generate calls/day/model). For production volumes, enable billing in Google AI Studio or switch models.
- Embedding model: `gemini-embedding-001` (task types: `RETRIEVAL_DOCUMENT` for ingest, `RETRIEVAL_QUERY` for search).
- Chat model: `gemini-3.5-flash-lite` for low latency.