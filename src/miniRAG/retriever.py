import json
from pathlib import Path

import ollama

from datasets import Dataset
from miniRAG.utils.log import Logger

logger = Logger()


class VectorDB:
    def __init__(self, db_path, embedding):
        self.db_path = Path(db_path)
        self.vector_db = []
        self.embedding = embedding

    def embed_text(self, text):
        return ollama.embed(model=self.embedding, input=text)["embeddings"][0]

    def initialize(self, corpus: Dataset):
        """
        Each element in the VECTOR_DB will be a tuple (chunk, embedding)
        The embedding is a list of floats, for example: [0.1, 0.04, -0.34, 0.21, ...]
        TODO: Use a more efficient vector database
        """
        self.vector_db = [
            (row["contents"], self.embed_text(row["contents"])) for row in corpus
        ]
        with open(self.db_path, "w") as f:
            json.dump(self.vector_db, f)
        logger.log(f"Vector database built with {len(self.vector_db)} entries")

    def load(self):
        if self.db_path.exists():
            with open(self.db_path, "r") as f:
                self.vector_db = json.load(f)
                logger.log(f"Vector database loaded with {len(self.vector_db)} entries")
        else:
            self.vector_db = []

    def __iter__(self):
        return iter(self.vector_db)

    def __len__(self):
        return len(self.vector_db)


class Retriever:
    def __init__(self, vector_db: VectorDB, top_n=3):
        self.vector_db = vector_db
        self.top_n = top_n

    def retrieve(self, user_query):
        """
        Given a user input, relevant splits are retrieved from storage using a Retriever.
        """

        def cosine_similarity(a, b):
            dot_product = sum([x * y for x, y in zip(a, b)])
            norm_a = sum([x**2 for x in a]) ** 0.5
            norm_b = sum([x**2 for x in b]) ** 0.5
            return dot_product / (norm_a * norm_b)

        similarities = []
        for chunk, embedding in self.vector_db:
            similarity = cosine_similarity(
                self.vector_db.embed_text(user_query), embedding
            )
            similarities.append((chunk, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[: self.top_n]
