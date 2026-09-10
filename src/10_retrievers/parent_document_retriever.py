from pathlib import Path
import sys

# -------------------------------------------------------------------
# Add the project root to Python's module search path.
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from langchain_classic.storage import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.helpers import print_separator, print_title

load_dotenv()


def main() -> None:
    """Demonstrates ParentDocumentRetriever."""

    print_title("Parent Document Retriever")

    loader = PyPDFLoader(
        PROJECT_ROOT / "data" / "input" / "sample.pdf"
    )

    documents = loader.load()

    print(f"Pages Loaded : {len(documents)}")

    print_separator()

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
    )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vector_store = FAISS.from_texts(
        texts=["dummy"],
        embedding=embeddings,
    )

    vector_store.delete(vector_store.index_to_docstore_id.values())

    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    retriever.add_documents(documents)

    query = "Explain the RAG pipeline."

    print(f"Query:\n{query}")

    print_separator()

    results = retriever.invoke(query)

    print(f"Retrieved Parent Documents : {len(results)}")

    print_separator()

    for index, document in enumerate(results, start=1):

        print(f"Parent Document {index}\n")

        print(document.page_content)

        print_separator()


if __name__ == "__main__":
    main()