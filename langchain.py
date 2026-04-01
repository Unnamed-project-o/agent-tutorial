import os

from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    ("system", "你是一位{role}，擅长用简洁易懂的方式解释复杂概念。"),
    ("human", "请解释一下：{concept}"),
])

messages = chat_template.format_messages(
    role="物理学教授",
    concept="量子纠缠"
)

llm = ChatTongyi(
    model="qwen-max",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

response = llm.invoke(messages)
print(response.content)