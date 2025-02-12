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

## Plan

Here're the plan for now: 

- Run our rag system on some benchmarks : 
    + [rgb](https://github.com/chen700564/RGB)
    + [flash rag](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets)
    + [crag](https://github.com/facebookresearch/CRAG)

- Optimize the rag system with different methods : [contextual retrieval](https://www.anthropic.com/news/contextual-retrieval), ...
- Make a web-ui for demo 


