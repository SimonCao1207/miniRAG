import os
from pathlib import Path


class Config:
    def __init__(self, config_dict: dict):
        self.embedding_model = config_dict["embedding_model"]
        self.model = config_dict["model"]
        self.model_id = _shortcut[config_dict["model"]]
        self.corpus_path = Path(config_dict["corpus_path"])
        self.index_path = Path(config_dict["index_path"])
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


_shortcut = {
    "gpt": "gpt-4o-mini",
    "llama": "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF",
}
