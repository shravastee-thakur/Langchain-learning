"""
File: openai_chat_model.py

Description
-----------
Demonstrates how to interact with an OpenAI Chat Model using LangChain.

Unlike creating the model manually with ChatOpenAI, this project reuses the
centralized LLM configuration provided by `llm_client.py`. This keeps all
configuration (provider selection, API keys, model name, etc.) in one place.

This lesson focuses only on sending prompts and receiving responses.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_client import get_llm


def chat_with_model() -> None:
    """
    Sends a simple prompt to the configured chat model and prints the response.

    This example demonstrates the most basic interaction with a LangChain
    chat model.
    """

    # Load the configured LLM from the shared project infrastructure.
    llm = get_llm()

    prompt = """
    Explain what LangChain is in exactly three concise bullet points.
    """

    print("=" * 80)
    print("USER PROMPT")
    print("=" * 80)
    print(prompt.strip())

    print("\nGenerating response...\n")

    # .invoke() is LangChain's standard way to get a single, full response
    response = llm.invoke(prompt)

    print("=" * 80)
    print("MODEL RESPONSE")
    print("=" * 80)

    # Most LangChain chat models return an AIMessage object.
    # The generated text is available inside the 'content' attribute.
    print(response.content)

    print("\n")
    print("=" * 80)
    print("Response Type")
    print("=" * 80)
    print(type(response))


if __name__ == "__main__":
    chat_with_model()