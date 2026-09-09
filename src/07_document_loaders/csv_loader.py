from pathlib import Path
import sys

# -------------------------------------------------------------------
# Add the project root to Python's module search path.
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_community.document_loaders import CSVLoader

from utils.helpers import print_separator, print_title

def main():
    print_title("CSV loader")

    file_path = PROJECT_ROOT / "data" / "input" / "employees.csv"
    loader = CSVLoader(file_path)

    documents = loader.load()

    print(f"Total Rows Loaded : {len(documents)}")

    print_separator()

    first_row = documents[0]

    print("Row Metadata:\n")
    print(first_row.metadata)

    print_separator()

    print("Row Content:\n")
    print(first_row.page_content)


if __name__ == "__main__":
    main()
