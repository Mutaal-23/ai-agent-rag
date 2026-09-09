from dotenv import load_dotenv
import os
import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from chromadb import PersistentClient

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

load_dotenv()

DATA_DIR = Path("data")
TEXT_EXTENSIONS = {".txt", ".json", ".pdf", ".html", ".htm"}


def read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text()
    if suffix == ".json":
        return json_to_text(path)
    if suffix == ".pdf":
        return pdf_to_text(path)
    if suffix in (".html", ".htm"):
        return BeautifulSoup(path.read_text(), "html.parser").get_text("\n")
    return ""


def json_to_text(path: Path) -> str:
    data = json.loads(path.read_text())
    return "\n".join(_flatten(data))


def _flatten(value, depth: int = 0) -> list[str]:
    lines = []
    pad = "  " * depth
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{pad}{k}:")
                lines += _flatten(v, depth + 1)
            else:
                lines.append(f"{pad}{k}: {v}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}{i + 1}.")
                lines += _flatten(item, depth + 1)
            else:
                lines.append(f"{pad}- {item}")
    else:
        lines.append(pad + str(value))
    return lines


def pdf_to_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def fetch_url(url: str) -> str:
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text("\n")


def load_all() -> list[tuple[str, str]]:
    docs = []
    for path in sorted(DATA_DIR.rglob("*")):
        if path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS:
            text = read_file(path).strip()
            if text:
                docs.append((text, str(path)))
    urls_file = DATA_DIR / "urls.txt"
    if urls_file.exists():
        for url in urls_file.read_text().splitlines():
            url = url.strip()
            if not url:
                continue
            try:
                print(f"  Fetching {url}")
                docs.append((fetch_url(url), url))
            except Exception as e:
                print(f"  skip {url}: {e}")
    return docs


if __name__ == "__main__":
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        task_type="RETRIEVAL_DOCUMENT",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    client = PersistentClient(path="./chroma_data")
    try:
        client.delete_collection("policies")
        print("Cleared old collection (no duplicates on re-run)")
    except Exception:
        print("No existing collection to clear")

    vectorstore = Chroma(
        client=client,
        collection_name="policies",
        embedding_function=embeddings,
    )

    total = 0
    for text, source in load_all():
        chunks = splitter.split_text(text)
        if not chunks:
            print(f"  {source}: no text found, skipping")
            continue
        vectorstore.add_texts(
            texts=chunks,
            metadatas=[{"source": source}] * len(chunks),
        )
        total += len(chunks)
        print(f"  {source}: {len(chunks)} chunks")

    print(f"Total chunks stored: {total}")