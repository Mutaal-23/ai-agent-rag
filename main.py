from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, AIMessage

from agent import agent, to_text


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    status: str
    response: str
    tools_used: list


app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = agent.invoke({"messages": [HumanMessage(content=req.message)]})

    final_message = result["messages"][-1]
    answer = to_text(final_message.content)

    tools_used = []
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            for call in msg.tool_calls:
                if call["name"] not in tools_used:
                    tools_used.append(call["name"])

    return ChatResponse(status="success", response=answer, tools_used=tools_used)