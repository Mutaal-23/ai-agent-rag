from dotenv import load_dotenv
import os
import sqlite3
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

from rag_tool import search_knowledge_base
from api_tool import get_weather

load_dotenv()


@tool
def knowledge_base(query: str) -> str:
    """Search the knowledge base for iPhone safety, policies, and rules.
    Use this for any question about the product guide, battery, or safety documentation."""
    return search_knowledge_base(query)


@tool
def weather(city: str) -> str:
    """Get the live weather in any city. Use this for any weather question."""
    return get_weather(city)


tools = [knowledge_base, weather]

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    max_output_tokens=1024,
)

llm_with_tools = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: AgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def to_text(content) -> str:
    if isinstance(content, str):
        return content
    parts = [b.get("text", "") for b in content
             if isinstance(b, dict) and b.get("type") == "text"]
    return "\n".join(parts)


def router(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


graph = StateGraph(AgentState)
graph.add_node("chatbot", chatbot)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "chatbot")
graph.add_conditional_edges("chatbot", router, {"tools": "tools", END: END})
graph.add_edge("tools", "chatbot")

conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
checkpointer = SqliteSaver(conn)

agent = graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    import sys
    from langchain_core.messages import HumanMessage

    question = sys.argv[1] if len(sys.argv) > 1 else "What is the policy on lasers?"
    session = sys.argv[2] if len(sys.argv) > 2 else "default"
    print(f"USER ({session}): {question}", flush=True)
    config = {"configurable": {"thread_id": session}}
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]}, config=config
    )
    print(f"AGENT: {to_text(result['messages'][-1].content)}", flush=True)