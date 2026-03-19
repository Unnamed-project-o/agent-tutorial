from openai import OpenAI
import os

client = OpenAI(
    # Please set the environment variable DASHSCOPE_API_KEY to your API key before running the code
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

messages = [{
             "role": "user", 
             "content": "你是谁"
            }]
completion = client.chat.completions.create(
    # You can replace it with other models, such as "qwen3-mini", "qwen3-medium", etc.
    model="qwen3-max",
    messages=messages,
    extra_body={"enable_thinking": True},
    stream=True
)
# Whether enter the answering phase
is_answering = False
for chunk in completion:
    delta = chunk.choices[0].delta
    if hasattr(delta, "content") and delta.content:
        if not is_answering:
            is_answering = True
        print(delta.content, end="", flush=True)