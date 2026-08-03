from openai import OpenAI
import os

client = OpenAI(
    # Please set the environment variable DASHSCOPE_API_KEY to your API key before running the code
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

messages = [
    {
        "role": "system", 
        "content": "你是一个Python高手, 精通数据分析和机器学习。请根据用户的需求，提供详细的代码示例和解释。"
    },
    {
        "role": "assistant",
        "content": "好的，请告诉我你需要什么样的帮助？"
    },
    {
        "role": "user", 
        "content": "请帮我写一段Python代码, 完成 csv 文件的读取和打印。不用有其他废话"
    },
]
completion = client.chat.completions.create(
    # You can replace it with other models, such as "qwen3-mini", "qwen3-medium", etc.
    model="qwen3.7-plus",
    messages=messages,
    extra_body={"enable_thinking": True},
    stream=True
)

# 打印结果
for chunk in completion:
    delta = chunk.choices[0].delta
    if hasattr(delta, "content") and delta.content:
        print(delta.content, end="", flush=True)
