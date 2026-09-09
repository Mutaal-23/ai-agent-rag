from dotenv import load_dotenv
import os
import re
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

from agent import agent as researcher_agent, to_text

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    max_output_tokens=1024,
)

MAX_DRAFTS = 2


class MultiAgentState(TypedDict, total=False):
    session_id: str
    question: str
    research: str
    draft: str
    score: int
    critique: str
    drafts: int
    research_tools: list


def researcher_node(state):
    config = {"configurable": {"thread_id": state.get("session_id", "default")}}
    result = researcher_agent.invoke(
        {"messages": [HumanMessage(content=state["question"])]}, config=config
    )

    tools = []
    for msg in result["messages"]:
        for call in getattr(msg, "tool_calls", []):
            if call["name"] not in tools:
                tools.append(call["name"])

    text = to_text(result["messages"][-1].content)
    return {"research": text, "research_tools": tools}


def writer_node(state):
    prompt = (
        "You are the Writer agent. Turn the raw research into a clear, friendly, "
        "complete answer for the user's question. Keep it under 8 sentences. "
        "Use only facts from the research.\n\n"
        f"QUESTION: {state['question']}\n\n"
        f"RESEARCH:\n{state.get('research', '')}\n\nANSWER:"
    )
    out = llm.invoke([HumanMessage(content=prompt)])
    return {"draft": to_text(out.content), "drafts": state.get("drafts", 0) + 1}


def critic_node(state):
    prompt = (
        "You are the Critic agent, a strict QA reviewer. "
        "Score the ANSWER against the RESEARCH (0-100) for correctness and completeness.\n\n"
        f"QUESTION: {state['question']}\n\n"
        f"RESEARCH:\n{state.get('research', '')}\n\n"
        f"ANSWER:\n{state.get('draft', '')}\n\n"
        'Reply ONLY as JSON: {"score": n, "issues": "short note"}'
    )
    out = to_text(llm.invoke([HumanMessage(content=prompt)]).content)

    match = re.search(r'"score":\s*(\d+)', out)
    score = int(match.group(1)) if match else 50
    issues = re.search(r'"issues":\s*"([^"]*)"', out)
    return {"score": score, "critique": issues.group(1) if issues else out[:120]}


def should_rewrite(state) -> bool:
    return state.get("score", 100) < 70 and state.get("drafts", 0) < MAX_DRAFTS


graph = StateGraph(MultiAgentState)
graph.add_node("researcher", researcher_node)
graph.add_node("writer", writer_node)
graph.add_node("critic", critic_node)
graph.add_edge(START, "researcher")
graph.add_edge("researcher", "writer")
graph.add_edge("writer", "critic")
graph.add_conditional_edges("critic", should_rewrite, {True: "writer", False: END})

multi_agent = graph.compile()


if __name__ == "__main__":
    import sys

    question = sys.argv[1] if len(sys.argv) > 1 else "What is the policy on lasers?"
    session = sys.argv[2] if len(sys.argv) > 2 else "default"
    print(f"USER: {question}", flush=True)
    state = multi_agent.invoke({"question": question, "session_id": session})
    print(f"[CRITIC] score={state['score']} issues={state['critique']}", flush=True)
    print(f"AGENT: {state['draft']}", flush=True)