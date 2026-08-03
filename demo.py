import os
import numpy as np
from openai import OpenAI
from llama_index.core.llms import CustomLLM
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLMMetadata

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex


from llama_index.core.llms import (
    CompletionResponse,
    CompletionResponseGen,
)

from llama_index.core import Settings
from llama_index.core.llms.callbacks import llm_completion_callback

api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

client = OpenAI(api_key=api_key, base_url=base_url)

class QwenLLM(CustomLLM):
    model: str = "qwen-plus"
    context_window: int = 131072
    num_output: int = 4096
    temperature: float = 0.7

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model,
        )
    
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: any) -> CompletionResponse:
        """执行大模型调用，接收输入prompt并返回模型输出"""
        try:
            # 调用通义千问API完成对话生成
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.num_output,
            )
            # 提取并返回模型生成的文本内容
            return CompletionResponse(
                text=response.choices[0].message.content.strip(),
                completion_gen=response,
            )
        except Exception as e:
            # 异常处理，抛出调用异常
            raise RuntimeError(f"调用千问大模型失败: {str(e)}") from e
    
    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: any) -> CompletionResponseGen:
        """执行大模型调用，接收输入prompt并返回模型输出"""
        try:
            # 调用通义千问API完成对话生成
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.num_output,
                stream=True,
            )
            # 提取并返回模型生成的文本内容
            for chunk in response:
                yield CompletionResponse(
                    text=chunk.choices[0].delta.content.strip(),
                    completion_gen=chunk,
                )
        except Exception as e:
            # 异常处理，抛出调用异常
            raise RuntimeError(f"调用千问大模型失败: {str(e)}") from e

class QwenEmbedding(BaseEmbedding):
    model: str = "text-embedding-v3"
    dimension: int = 1024

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @classmethod
    def class_name(cls) -> str:
        return "QwenEmbedding"
    
    async def _aget_query_embedding(self, query: str) -> np.ndarray:
        """将查询字符串转换为嵌入向量"""
        return self._embed_query(query)
    
    async def _aget_text_embedding(self, text: str) -> np.ndarray:
        """将文本字符串转换为嵌入向量"""
        return self._embed_query(text)

    def _embed_query(self, query: str) -> np.ndarray:
        """将查询字符串转换为嵌入向量"""
        try:
            # 调用通义千问API完成嵌入生成
            response = client.embeddings.create(
                model=self.model,
                input=query,
            )
            # 提取并返回模型生成的嵌入向量
            return response.data[0].embedding
        except Exception as e:
            # 异常处理，抛出调用异常
            raise RuntimeError(f"调用千问嵌入模型失败: {str(e)}") from e
    
    def _get_text_embedding(self, text: str) -> np.ndarray:
        """将文本字符串转换为嵌入向量"""
        return self._embed_query(text)
    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """将查询字符串转换为嵌入向量"""
        return self._embed_query(query)
    


Settings.llm = QwenLLM()
Settings.embed_model = QwenEmbedding()

questions = "DS是什么？怎么学？"
response = Settings.llm.complete(questions)
print(response.text)

resource_dir = os.path.join(os.path.dirname(__file__), "resource")
docs = SimpleDirectoryReader(resource_dir).load_data()

index = VectorStoreIndex.from_documents(docs)

query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query(questions)
print(response)
