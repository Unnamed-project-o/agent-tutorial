from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

messages = [{
             "role": "user", 
             "content": "你是谁"
            }]
completion = client.chat.completions.create(
    model="qwen3.7-flash-2026-07-15",
    messages=messages,
    extra_body={"enable_thinking": True},
    stream=True
)

is_answering = False
print("\n" + "=" * 20 + "Thinking Process" + "=" * 20)
for chunk in completion:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    if delta is None:
        continue
    if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
        if not is_answering:
            print(delta.reasoning_content, end="", flush=True)
    if hasattr(delta, "content") and delta.content:
        if not is_answering:
            print("\n" + "=" * 20 + "Complete Response" + "=" * 20)
            is_answering = True
        print(delta.content, end="", flush=True)