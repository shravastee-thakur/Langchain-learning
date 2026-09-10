import os
import shutil
from pathlib import Path
import sys

# Disable Chroma telemetry to prevent hanging issues during vector additions
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.helpers import print_separator, print_title

load_dotenv()


def main() -> None:
    print_title("Chroma Vector Store")


    file_path = PROJECT_ROOT / "data" / "input" / "sample.txt"
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"Chunks Created : {len(chunks)}")
    print_separator()


    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    chroma_db_path = PROJECT_ROOT / "data" / "chroma_db"

    if chroma_db_path.exists():
        shutil.rmtree(chroma_db_path)


    print("Adding documents to Chroma...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="langchain_demo",
        persist_directory=str(chroma_db_path),
    )
    print("Documents successfully stored.")
    print_separator()


    print(f"Document Count in Collection : {vector_store._collection.count()}")
    print_separator()


    query = "What is Generative AI?"
    print(f"Query: {query}")
    print("Running similarity search...")

    results = vector_store.similarity_search(query=query, k=2)
    print(f"Retrieved {len(results)} document(s).\n")
    print_separator()

    for idx, doc in enumerate(results, start=1):
        print(f"Result {idx}:\n{doc.page_content}\n")
        print_separator()


if __name__ == "__main__":
    main()