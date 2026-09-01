from pathlib import Path
import sys

# -------------------------------------------------------------------
# Add the project root to Python's module search path.
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.prompts import ChatPromptTemplate

from llm_client import get_llm
from utils.helpers import print_separator, print_title

# JSON Schema
movie_review_schema = {
    "title": "MovieReview",
    "description": "Schema for a movie review.",
    "type": "object",
    "properties": {
        "movie_name": {
            "type": "string",
            "description": "Name of the movie."
        },
        "rating": {
            "type": "number",
            "description": "Rating out of 10."
        },
        "summary": {
            "type": "string",
            "description": "Short review of the movie."
        },
        "genres": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of movie genres."
        }
    },
    "required": ["movie_name", "rating", "summary", "genres"]
}

def main():
    print_title("Structured Output")

    llm = get_llm().with_structured_output(movie_review_schema)

    response = llm.invoke( """
        Review the movie 'Interstellar'.

        Keep the summary under 60 words.
        """)

    print("Structured Response:\n")
    print(response)

    print_separator()

    print("Accessing Individual Fields:\n")

    print(f"Movie Name : {response['movie_name']}")
    print(f"Rating     : {response['rating']}")
    print(f"Genres     : {response['genres']}")
    print(f"Summary    : {response['summary']}")


if __name__ == "__main__":
    main()

    