# miniRAG

An implementation of a simple RAG system, similar to Langchain. miniRAG tries to be small, clean, interpretable and educational.


## Download ollama and models
Open a terminal and run the following command to download the required models
```
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
```

## Install and Run

- Use [uv](https://docs.astral.sh/uv/pip/environments/) to create virtual env

```
uv pip install -e .
```

## Run Test
```
pytest
```
## Feature

### FlashRAG
- Provide a toy index (`/datasets/flashRAG/index/general_knowledge_v1.index`), built using E5-base-v2 and the toy corpus from [flashRAG](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets) 
- To evaluate the method, run the toy benchmark with: 
```
pytest -s tests/test_flashRAG.py
```

