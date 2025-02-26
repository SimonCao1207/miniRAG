import json
import os
from pathlib import Path

import dspy
import pytest

from miniRAG.config import Config
from miniRAG.rag import RAG
from miniRAG.retriever import Retriever, VectorDB, load_corpus

workspace = Path(__file__).resolve().parent.parent
if not os.path.exists("tmp"):
    os.mkdir("tmp")

config_dict = {
    "model": "llama",
    "embedding_model": "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf",
    "corpus_path": workspace / "datasets/flashRAG/general_knowledge.jsonl",
    "index_path": workspace / "datasets/flashRAG/index/general_knowledge_v1.index",
    "test_file": workspace / "datasets/flashRAG/test.jsonl",
}


@pytest.fixture(scope="session")
def setup_rag():
    config = Config(config_dict)
    corpus = load_corpus(config.corpus_path)
    vector_db = VectorDB(config.index_path, config.embedding_model, is_split=False)
    if not config.index_path.exists():
        vector_db.build_index(corpus)
    vector_db.load()
    retriever = Retriever(vector_db=vector_db)
    model_id = config.model_id
    if config.model == "gpt":
        model = dspy.LM(model_id, api_key=os.getenv("OPENAI_API_KEY"))
    elif config.model == "llama":
        model = dspy.LM(model_id, api_base="http://localhost:11434", api_key="")
    rag = RAG(model, retriever)
    return rag


def read_jsonl(file_path):
    data_list = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            entry = json.loads(line.strip())
            data_list.append(entry)
    return data_list


def generate_test_function(query, answer):
    def test_function(setup_rag):
        rag = setup_rag
        response = rag.generate(query)
        # TODO: get structure answer from response
        assert response in answer

    return test_function


test_data = read_jsonl(config_dict["test_file"])
for i, item in enumerate(test_data):
    query, answer = item["question"], item["golden_answers"]
    test_func = generate_test_function(query, answer)
    globals()[f"Case_{i}"] = test_func


if __name__ == "__main__":
    dct = read_jsonl(config_dict["test_file"])
    print(dct)
