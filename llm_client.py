import json
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()


def get_llm():
    """ Loads config and returns a LangChain Groq model. """

    with open("config.json", "r") as file:
        config = json.load(file)

        """
        open("config.json", "r") as file: This tells Python to find the file named config.json and open it in "r" (read-only) mode. It temporarily names this opened file object file.

        config = json.load(file): This takes the raw text from the JSON file and translates it into a Python dictionary, saving it to the variable config. So {"provider": "groq"} in JSON becomes config["provider"] in Python.
        
        """

        return ChatGroq(
            model=config["groq"]["model"],
            temperature=config["groq"]["temperature"],
            max_tokens=config["groq"]["max_tokens"],
            api_key=os.getenv("GROQ_API_KEY"),
        )
    
   