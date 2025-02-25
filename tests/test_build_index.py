import os
import shutil
import time
from pathlib import Path

from miniRAG.config import Config
from miniRAG.retriever import VectorDB, load_corpus

workspace = Path(__file__).resolve().parent.parent

if os.path.exists("tmp"):
    shutil.rmtree("tmp")
os.mkdir("tmp")

config_dict = {
    "embedding_model": "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf",
    "model": "llama",
    "corpus_path": workspace / "datasets/flashRAG/general_knowledge.jsonl",
    "index_path": workspace / "tmp/vector_db_flashRAG.index",
    "chunk_size": 512,
}


def test_build_index_time():
    config = Config(config_dict)
    corpus = load_corpus(config.corpus_path)
    vector_db = VectorDB(config.index_path, config.embedding_model, is_split=False)

    start_time = time.perf_counter()
    vector_db.build_index(corpus)
    end_time = time.perf_counter()

    exec_time = end_time - start_time
    print(f"Building index took {exec_time:.2f} seconds")

    # Currently, the index building process is not optimized and takes a long time, around 1.5 hours.
    assert exec_time < 3600, "Index building took too long"


if __name__ == "__main__":
    test_build_index_time()
