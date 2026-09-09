from pathlib import Path
import sys

# -------------------------------------------------------------------
# Add the project root to Python's module search path.
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_community.document_loaders import WebBaseLoader

from utils.helpers import print_separator, print_title


def main() -> None:
    """Demonstrates WebBaseLoader."""

    print_title("Web Loader")

    loader = WebBaseLoader(
        web_paths=(
            "https://python.langchain.com/docs/introduction/",
        )
    )

    documents = loader.load()

    print(f"Total Documents Loaded : {len(documents)}")

    print_separator()

    document = documents[0]

    print("Document Metadata:\n")
    print(document.metadata)

    print_separator()

    print("Document Content (First 1000 Characters):\n")
    print(document.page_content[:1000])


if __name__ == "__main__":
    main()
