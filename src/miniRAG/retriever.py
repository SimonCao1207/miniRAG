import json
from pathlib import Path
from typing import List

import ollama

import datasets
from datasets import Dataset
from miniRAG.documents import Document
from miniRAG.utils.log import Logger

logger = Logger()


def load_corpus(corpus_path: Path) -> List[Document]:
    logger.log("Loading the corpus...")
    documents = []
    if corpus_path.suffix == ".jsonl":  # FlashRAG corpus
        corpus: Dataset = datasets.load_dataset(  # type: ignore
            "json", data_files=str(corpus_path), split="train"
        )
        for row in corpus:
            documents.append(
                Document(
                    page_content=row["contents"],
                    metadata={"id": str(row["id"]), "title": row["title"]},
                )
            )
    else:
        with open(corpus_path, "r") as file:
            data = file.read()
        documents.append(Document(page_content=data))
    return documents


class TextSplitter:
    """
    This Splitter will split our documents into chunks of chunk_size characters
    with chunk_overlap characters of overlap between chunks. The overlap
    helps mitigate the possibility of separating a statement from important
    context related to it.
    """

    def __init__(self, chunk_size, chunk_overlap):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, docs: List[Document]) -> List[Document]:
        split_docs = []
        for doc in docs:
            text = doc.page_content
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk = text[start:end]
                split_docs.append(Document(page_content=chunk, metadata=doc.metadata))
                start += self.chunk_size - self.chunk_overlap
        return split_docs


class VectorDB:
    def __init__(self, db_path, embedding, is_split=False):
        self.db_path = Path(db_path)
        self.vector_db = []
        self.embedding = embedding
        self.is_split = is_split

    def embed_text(self, text):
        return ollama.embed(model=self.embedding, input=text)["embeddings"][0]

    def build_index(self, corpus: List[Document]):
        """
        This is where the indexing(splitting) happens.
        Currently, each element in the VECTOR_DB will be a tuple (chunk, embedding)
        TODO: Use a more efficient vector database, with better indexing approach.
        """
        split_docs: List[Document] = corpus
        if self.is_split:
            splitter = TextSplitter(chunk_size=512, chunk_overlap=50)
            split_docs = splitter.split_documents(corpus)

        # This might be to slow for 15000 split_docs
        self.vector_db = [
            (doc.page_content, self.embed_text(doc.page_content)) for doc in split_docs
        ]
        self.save_index()

    def save_index(self):
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
