from pathlib import Path
import sys

# -------------------------------------------------------------------
# Add the project root to Python's module search path.
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.helpers import print_separator, print_title

load_dotenv()


def main() -> None:
    """Demonstrates an MMR Retriever."""

    print_title("MMR Retriever")

    loader = TextLoader(
        PROJECT_ROOT / "data" / "input" / "sample.txt"
    )

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
    )

    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 2,
            "fetch_k": 4,
        },
    )

    query = "Explain Artificial Intelligence."

    print(f"Query:\n{query}")

    print_separator()

    documents = retriever.invoke(query)

    for index, document in enumerate(documents, start=1):

        print(f"Retrieved Document {index}\n")

        print(document.page_content)

        print_separator()


if __name__ == "__main__":
    main()