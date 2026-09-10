import os
from pathlib import Path
import sys

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.helpers import print_separator, print_title

load_dotenv()

def main():
    print_title("Similarity Search")

    loader = TextLoader(PROJECT_ROOT / "data" / "input" / "sample.txt")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 50
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings,
        )
    query = "Explain Large Language Models."

    print(f"User Query:\n{query}")

    print_separator()

    results = vector_store.similarity_search(
        query=query,
        k=3,
    )

    for index, document in enumerate(results, start=1):

        print(f"Retrieved Chunk {index}")

        print()

        print(document.page_content)

        print_separator()


if __name__ == "__main__":
    main()

    