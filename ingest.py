from dotenv import load_dotenv
import os 
from pathlib import Path



load_dotenv()
key = os.getenv("GEMINI_API_KEY")

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_chroma import Chroma

text = Path("data/file.txt").read_text()

print(f"Loaded text: {len(text)} characters")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

chunks = splitter.split_text(text)

print(f"Created {len(chunks)} chunks")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",       # ← current model name
    task_type="RETRIEVAL_DOCUMENT",            # ← "I'm embedding docs to store"
    google_api_key=os.getenv("GEMINI_API_KEY")
)

vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    collection_name="policies",
    persist_directory="./chroma_data"
)

print("Documents stored in ChromaDB!")