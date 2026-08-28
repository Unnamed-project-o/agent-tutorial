# LlamaIndex + Ragas 小型 RAG 系统

Notebook：`LlamaIndex_Ragas_小型RAG系统.ipynb`

## 运行

```bash
conda activate agent-base
python -m ipykernel install --user --name agent-base --display-name "agent-base"
```

使用现有的 Jupyter、VS Code 或 PyCharm 打开 Notebook，选择 `agent-base` Kernel，然后从头执行所有单元。启动时请把工作目录保持在本文件夹，使 Notebook 能找到同级的 `knowledge_base` 目录。当前 `agent-base` 未安装 JupyterLab；如需从该环境启动，请先运行 `conda install -c conda-forge jupyterlab -y`。

知识库由 6 份企业制度 Word 文件组成，LlamaIndex 使用 `SimpleDirectoryReader + DocxReader` 直接读取。原始 Word 文件没有 `doc_id`，评估时使用加载器生成的文件名 Metadata 追踪来源。

主实验使用 DashScope 进行 Embedding 与答案生成，使用 Ragas 的确定性指标进行自动评估。需要提前配置 `DASHSCOPE_API_KEY`。Notebook 末尾的 LLM Judge 默认关闭。
