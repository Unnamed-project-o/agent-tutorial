import os

from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate

few_shot_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个情绪表达助手。"),
    ("human", "示例：\n情绪: 开心\n表达: 我今天非常开心！\n\n情绪: 难过\n表达: 我感到有些难过..."),
    ("human", "现在请表达这个情绪: {emotion}")
])

messages = few_shot_template.format_messages(emotion="兴奋")

llm = ChatTongyi(
    model="qwen-max",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

response = llm.invoke(messages)
print(response.content)