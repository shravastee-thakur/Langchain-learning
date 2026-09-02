from pathlib import Path
import sys

# -------------------------------------------------------------------
# Add the project root to Python's module search path.
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from langchain_classic.memory import ConversationBufferWindowMemory

from utils.helpers import print_separator, print_title

def main():
    print_title("Conversation Buffer Window Memory")

    memory = ConversationBufferWindowMemory(
        k=2,
        return_messages=True
        )

    conversations = [
        ("Hi!", "Hello!"),
        ("My name is Vikash.", "Nice to meet you, Vikash."),
        ("I am learning LangChain.", "Great choice!"),
        ("Tell me about RAG.", "RAG combines retrieval with generation."),
        ("What are AI Agents?", "AI Agents can plan, reason, and use tools."),
    ]

    for human, ai in conversations:
        memory.save_context(
            {"input": human},
            {"output": ai}
        )


    history = memory.load_memory_variables({})

    print("Conversation Stored In Memory:\n")

    for index, message in enumerate(history["history"], start=1):
        print(f"Message {index}")
        print(f"Type    : {message.type}")
        print(f"Content : {message.content}")
        print_separator()


if __name__ == "__main__":
    main()