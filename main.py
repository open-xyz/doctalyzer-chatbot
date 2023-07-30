import textbase
from textbase.message import Message
from textbase import models
import os
from typing import List
# Load your OpenAI API key
models.OpenAI.api_key = ""
# or from environment variable:
models.OpenAI.api_key = os.getenv("OPENAI_API_KEY")

# Prompt for GPT-3.5 Turbo
SYSTEM_PROMPT = """I want you to be an AI doctor and tell me how you can help me and I don't want you to respond to irrelevant to medical terms. Please understand that you are a AI doctor and can help with anything realted to medical. Also, Please be specific and explain it like a five year child. Also please present the data systematically points by points
"""


@textbase.chatbot("talking-bot")
def on_message(message_history: List[Message], state: dict = None):
    """Your chatbot logic here
    message_history: List of user messages
    state: A dictionary to store any stateful information

    Return a string with the bot_response or a tuple of (bot_response: str, new_state: dict)
    """

    if state is None or "counter" not in state:
        state = {"counter": 0}
    else:
        state["counter"] += 1

    # # Generate GPT-3.5 Turbo response
    bot_response = models.OpenAI.generate(
        system_prompt=SYSTEM_PROMPT,
        message_history=message_history,
        model="gpt-3.5-turbo",
    )

    return bot_response, state
