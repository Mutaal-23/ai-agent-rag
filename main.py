from fastapi import FastAPI
from pydantic import BaseModel

from multi_agent import multi_agent


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    status: str
    session_id: str
    response: str
    tools_used: list
    agents_used: list[str]
    critic_note: str | None = None


app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or "default"

    state = multi_agent.invoke(
        {"question": req.message, "session_id": session_id}
    )

    return ChatResponse(
        status="success",
        session_id=session_id,
        response=state.get("draft") or "No answer generated.",
        tools_used=state.get("research_tools", []),
        agents_used=["researcher", "writer", "critic"],
        critic_note=state.get("critique"),
    )