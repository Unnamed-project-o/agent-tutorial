import os

from langchain_community.chat_models import ChatTongyi

llm = ChatTongyi(
    model="qwen-max",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

response = llm.invoke("你好，请用一句话介绍自己")
print(response.content)