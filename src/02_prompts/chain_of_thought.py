
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.prompts import ChatPromptTemplate

from llm_client import get_llm
from utils.helpers import print_separator, print_title

def main():
    print_title("Chain-Of-Thought-Prompting")
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
                """
You are an experienced insurance claims analyst.

Think through the problem step by step before reaching your conclusion.
Finally, provide:
1. Your reasoning
2. Your final decision
""",),
            ("human",
                """
A vehicle was insured on January 1.

The policy covers accidents occurring after the policy start date.

The customer reports:
- Accident Date: January 15
- Claim Filed: January 18
- Premium Status: Paid
- Police Report: Available
- Estimated Repair Cost: $4,800

Should this claim be approved? Explain your reasoning step by step.
""",
            ),
        ]
    )

    messages = prompt.invoke({})

    print("Formated promot:\n")

    for message in messages.messages:
        print(f"{message.type.upper()}:")
        print(message.content)
        print()

    print_separator()

    response = llm.invoke(messages)

    print("LLM Response:\n")
    print(response.content)


if __name__ == "__main__":
    main()