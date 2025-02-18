import os
from pathlib import Path

import pytest

from miniRAG.config import Config
from miniRAG.models import OllamaModel, OpenAIServerModel
from miniRAG.rag import RAG
from miniRAG.retriever import Retriever, VectorDB, load_corpus

workspace = Path(__file__).resolve().parent.parent

if not os.path.exists(workspace / "tmp"):
    os.mkdir(workspace / "tmp")

config_dict = {
    "model": "llama",
    "embedding_model": "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf",
    "corpus_path": workspace / "datasets/cat-facts.txt",
    "index_path": workspace / "tmp/vector_db.json",
}


@pytest.fixture
def setup_rag():
    config = Config(config_dict)
    corpus = load_corpus(config.corpus_path)
    vector_db = VectorDB(config.index_path, config.embedding_model, is_split=True)
    if not config.index_path.exists():
        vector_db.build_index(corpus)
    vector_db.load()
    retriever = Retriever(vector_db=vector_db)
    model_id = config.model_id
    if config.model == "gpt":
        model = OpenAIServerModel(
            model_id=model_id,
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base="https://api.openai.com/v1",
        )
    elif config.model == "llama":
        model = OllamaModel(model_id)
    rag = RAG(model, retriever)
    return rag


def test_cat_facts_split(setup_rag):
    rag = setup_rag
    input_query = "What is the cat top speed? Answer short without explanation."
    response = rag.generate(input_query)
    assert "31 mph" in response
