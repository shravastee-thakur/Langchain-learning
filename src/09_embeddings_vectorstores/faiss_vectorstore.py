from pathlib import Path
import sys
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# -------------------------------------------------------------------
# Add the project root to Python's module search path.
# -------------------------------------------------------------------
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


def main() -> None:
    """Demonstrates FAISS Vector Store."""

    print_title("FAISS Vector Store")

    loader = TextLoader(
        PROJECT_ROOT / "data" / "input" / "sample.txt"
    )

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    print(f"Documents Indexed : {len(chunks)}")

    print_separator()

    query = "What is Machine Learning?"

    results = vector_store.similarity_search(
        query=query,
        k=2,
    )

    print(f"Query:\n{query}")

    print_separator()

    for index, document in enumerate(results, start=1):

        print(f"Result {index}\n")

        print(document.page_content)

        print_separator()


if __name__ == "__main__":
    main()