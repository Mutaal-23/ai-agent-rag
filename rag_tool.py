from dotenv import load_dotenv
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def search_knowledge_base(query: str) -> str:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        task_type="RETRIEVAL_QUERY",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    db = Chroma(
        persist_directory="./chroma_data",
        collection_name="policies",
        embedding_function=embeddings
    )

    docs = db.similarity_search(query, k=3)

    result = ""
    for i, doc in enumerate(docs):
        if i > 0:
            result += "\n\n---\n\n"
        result += doc.page_content

    return result

if __name__ == "__main__":
    print(search_knowledge_base("What are the warnings about batteries?"))