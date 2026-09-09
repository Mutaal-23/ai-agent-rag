from dotenv import load_dotenv
import os
import re

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from chromadb import PersistentClient

load_dotenv()

COLLECTION = "policies"
K = 3


def search_knowledge_base(query: str, k: int = K) -> str:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        task_type="RETRIEVAL_QUERY",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
    client = PersistentClient(path="./chroma_data")
    collection = client.get_or_create_collection(COLLECTION)
    data = collection.get()

    vector_ids = vector_rank(collection, embeddings, query, k)
    keyword_ids = keyword_rank(data, query, k)
    fused_ids = rrf([vector_ids, keyword_ids])[:k]

    id_to_text = dict(zip(data["ids"], data["documents"]))
    parts = [id_to_text[i] for i in fused_ids if i in id_to_text]
    return "\n\n---\n\n".join(parts)


def vector_rank(collection, embeddings, query: str, k: int) -> list[str]:
    query_embedding = embeddings.embed_query(query)
    result = collection.query(query_embeddings=[query_embedding], n_results=k)
    return result["ids"][0]


def _tokens(text: str) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words}


def keyword_rank(data, query: str, k: int) -> list[str]:
    query_tokens = _tokens(query)
    scores = []
    for doc_id, text in zip(data["ids"], data["documents"]):
        overlap = len(query_tokens & _tokens(text))
        scores.append((overlap, doc_id))
    scores.sort(key=lambda item: -item[0])
    return [doc_id for overlap, doc_id in scores[:k] if overlap > 0]


def rrf(rankings: list[list[str]], const: int = 60) -> list[str]:
    scores = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0) + 1.0 / (const + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


if __name__ == "__main__":
    query = "What is the policy on lasers?"
    print(search_knowledge_base(query))