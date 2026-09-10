from pathlib import Path
import sys

# -------------------------------------------------------------------
# Add the project root to Python's module search path.
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from utils.helpers import print_separator, print_title

load_dotenv()

def main():
    print_title("Text Embeddings")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    text = "LangChain makes it easy to build applications powered by LLMs."
    vector = embeddings.embed_query(text)

    print("Input Text:\n")
    print(text)

    print_separator()

    print(f"Embedding Dimension : {len(vector)}")

    print_separator()

    print("First 10 Vector Values:\n")
    print(vector[:10])


if __name__ == "__main__":
    main()