import os

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="glm-5.1",
    api_key="sk-idx6rzJH2-anYC-n0h40aQ",
    # ChatOpenAI 会自动在 base_url 后追加 /chat/completions。
    base_url="http://113.46.219.251:8080/v1",
    temperature=1.0,
    streaming=True,
    model_kwargs={
        "extra_body": {
            "enable_thinking": True,
        }
    },
)

messages = [HumanMessage(content="你是谁")]

for chunk in llm.stream(messages):
    if chunk.content:
        print(chunk.content, end="", flush=True)

print()
