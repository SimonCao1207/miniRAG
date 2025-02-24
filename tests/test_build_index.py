import os
import time
from pathlib import Path

from miniRAG.config import Config
from miniRAG.retriever import VectorDB, load_corpus

workspace = Path(__file__).resolve().parent.parent
if not os.path.exists("tmp"):
    os.mkdir("tmp")

config_dict = {
    "embedding_model": "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf",
    "model": "llama",
    "corpus_path": workspace / "datasets/flashRAG/general_knowledge.jsonl",
    "index_path": workspace / "tmp/vector_db_flashRAG.json",
}


def test_build_index_time():
    config = Config(config_dict)
    corpus = load_corpus(config.corpus_path)
    vector_db = VectorDB(config.index_path, config.embedding_model, is_split=True)

    start_time = time.perf_counter()
    vector_db.build_index(corpus)
    end_time = time.perf_counter()

    exec_time = end_time - start_time
    print(f"Building index took {exec_time:.2f} seconds")
    assert exec_time < 60, "Index building took too long"


if __name__ == "__main__":
    test_build_index_time()
