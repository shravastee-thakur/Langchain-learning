from pathlib import Path
import sys

# -------------------------------------------------------------------
# Add project root
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from llm_client import get_llm
from utils.helpers import print_separator, print_title


def main():

    print_title("Router Chain")

    llm = get_llm()

    science_chain = (
        PromptTemplate.from_template(
            "Explain the following science concept:\n\n{query}"
        )
        | llm
        | StrOutputParser()
    )

    coding_chain = (
        PromptTemplate.from_template(
            "Answer the following programming question:\n\n{query}"
        )
        | llm
        | StrOutputParser()
    )

    general_chain = (
        PromptTemplate.from_template(
            "Answer the following question:\n\n{query}"
        )
        | llm
        | StrOutputParser()
    )

    def route(info):
        query = info["query"].lower()

        if "python" in query or "code" in query:
            print("Routing to: Coding Chain\n")
            return coding_chain

        elif "llm" in query or "rag" in query or "embedding" in query or "vector" in query:
            print("Routing to: AI Concepts Chain\n")
            return science_chain

        else:
            print("Routing to: General Chain\n")
            return general_chain

    chain = RunnableLambda(route)

    query = "What are Vector Embeddings?"

    print(f"Query:\n{query}")

    print_separator()

    response = chain.invoke({"query": query})

    print(response)


if __name__ == "__main__":
    main()