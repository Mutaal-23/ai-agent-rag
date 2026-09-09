from dotenv import load_dotenv
import os
import json
import sys

import rag_tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()

TEST_SET = [
    {
        "question": "What is the policy on lasers?",
        "topic": "lasers",
    },
    {
        "question": "How many vacation days do employees get?",
        "topic": "vacation policy (JSON)",
    },
    {
        "question": "What should I do if the laser system is disabled?",
        "topic": "laser disabling",
    },
]

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    max_output_tokens=400,
)


def _to_text(content) -> str:
    if isinstance(content, str):
        return content
    text = ""
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                text += block.get("text", "")
        elif getattr(block, "type", None) == "text":
            text += getattr(block, "text", "")
    return text


def answer_from_context(question: str, context: str) -> str:
    prompts = [
        (
            "Answer the question using ONLY the context below. "
            "If the answer is not in the context, say 'Not in context'.\n"
            f"CONTEXT:\n{context[:2000]}\n\n"
            f"QUESTION: {question}\n\nAnswer concisely."
        ),
        (
            "The context below DOES contain the answer. "
            "Find it and quote the relevant part, then answer in one sentence.\n"
            f"CONTEXT:\n{context[:2000]}\n\n"
            f"QUESTION: {question}\n\nAnswer directly."
        ),
    ]
    for i, prompt in enumerate(prompts):
        out = llm.invoke([HumanMessage(content=prompt)])
        answer = _to_text(out.content).strip()
        if i == 0 and "not in context" not in answer.lower():
            return answer
    return answer


def judge(question: str, context: str, answer: str) -> dict | None:
    out = llm.invoke(
        [
            HumanMessage(
                content=(
                    "Score this RAG response with STRICT rules.\n"
                    f"QUESTION: {question}\n"
                    f"CONTEXT: {context[:1600]}\n"
                    f"ANSWER: {answer}\n\n"
                    'Return ONLY JSON: {"faithfulness": x, "response_relevancy": y, "context_precision": z}\n'
                    "- faithfulness: claims in ANSWER supported by CONTEXT? 0.0 if the answer "
                    "contains claims absent from the context.\n"
                    "- response_relevancy: does ANSWER give the requested information? "
                    "0.0 if it fails to answer (e.g. says 'not in context' while the context "
                    "actually contains the answer), 0.5 for partial, 1.0 for complete.\n"
                    "- context_precision: is CONTEXT relevant to QUESTION? 0.0 if unusable.\n"
                    "Use the full 0.0-1.0 range. Do not default to 1.0."
                )
            )
        ]
    )
    text = _to_text(out.content)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    return json.loads(text[start : end + 1])


def run(limit: int | None = None) -> None:
    tests = TEST_SET[:limit] if limit else TEST_SET
    totals = {"faithfulness": 0.0, "response_relevancy": 0.0, "context_precision": 0.0}
    print(f"{'TOPIC':<32}{'F':<8}{'R':<8}{'C':<8}")
    print("-" * 56)
    for test in tests:
        context = rag_tool.search_knowledge_base(test["question"])
        answer = answer_from_context(test["question"], context)
        scores = judge(test["question"], context, answer) or {}
        for key in totals:
            totals[key] += scores.get(key, 0.0)
        print(
            f"{test['topic'][:32]:<32}"
            f"{scores.get('faithfulness', 0.0):<8.2f}"
            f"{scores.get('response_relevancy', 0.0):<8.2f}"
            f"{scores.get('context_precision', 0.0):<8.2f}"
        )
        print(f"  Q: {test['question']}")
        print(f"  A: {answer[:110]}")
        print()
    count = len(tests)
    print("-" * 56)
    print("AVERAGES:", {k: round(v / count, 2) for k, v in totals.items()})


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run(limit)