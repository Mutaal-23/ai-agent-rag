# AI Agent with RAG + Live API Tools + Multi-Agent Memory

A production-style agentic AI system built with **LangGraph, LangChain, Google Gemini, and FastAPI**. It answers user questions by combining:

1. **Knowledge-base search (RAG)** — hybrid retrieval (semantic + keyword) from a local ChromaDB store
2. **Live API calls** — fetches real-time weather from the Open-Meteo API
3. **Session memory** — every conversation stored in SQLite, survives server restarts
4. **Multi-agent orchestration** — Researcher → Writer → Critic pipeline with automated review
5. **Plain reasoning** — answers general questions directly, no tools needed

The system decides on its own when to search the knowledge base, call the API, or just answer, and a QA critic verifies every response before it is delivered.

## Architecture

```
User question
      │
      ▼
POST /api/v1/chat  (FastAPI)
      │
      ▼
┌───────────────────────────────────────────────────┐
│  Multi-Agent Graph (LangGraph)                    │
│  ┌────────────┐    ┌────────┐    ┌─────────────┐  │
│  │ Researcher │───▶│ Writer │───▶│   Critic    │  │
│  │ (ReAct +   │    │ polish│    │ QA score 0-100│ │
│  │  tools +   │    │ draft │    └──────┬──────┘   │
│  │  memory)   │    └────────┘         │           │
│  └────────────┘                        │ score<70 │
│                     ┌──────────────────┘  & retries
│                     ▼                      left       │
│                back to Writer 🔄                      │
└───────────────────────────────────────────────────┘
      │
      ▼
{ "status", "session_id", "response", "tools_used",
  "agents_used", "critic_note" }
```

Files:
- `ingest.py` — loads `.txt`, `.json`, `.pdf`, `.html`, and web URLs (`data/urls.txt`) into ChromaDB
- `rag_tool.py` — hybrid retrieval (vector + keyword, fused with Reciprocal Rank Fusion)
- `api_tool.py` — live weather via Open-Meteo
- `agent.py` — the ReAct agent with SQLite session memory
- `multi_agent.py` — Researcher → Writer → Critic orchestration
- `eval.py` — LLM-judge evaluation (faithfulness / relevancy / context precision)
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

# 4. Put your knowledge documents in data/
#    (.txt, .json, .pdf, .html — or list web pages in data/urls.txt, one per line)

# 5. Build the knowledge base
python ingest.py

# 6. Optional: run the evaluation report
python eval.py

# 7. Start the server
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
{
  "message": "What is the policy on lasers?",
  "session_id": "user-123"
}
```

`session_id` is optional but recommended — pass the same id to keep a conversation's memory across turns. If omitted, it defaults to `"default"`.

Response:
```json
{
  "status": "success",
  "session_id": "user-123",
  "response": "According to the safety guidelines...",
  "tools_used": ["knowledge_base"],
  "agents_used": ["researcher", "writer", "critic"],
  "critic_note": "All facts verified."
}
```

- `tools_used` — which tools the researcher executed: `knowledge_base`, `weather`, or `[]` (answered directly)
- `agents_used` — the multi-agent pipeline that ran
- `critic_note` — QA feedback (and when the critic scores < 70 the writer auto-rewrites once)

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