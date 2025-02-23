import json
import time
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
    def __init__(self):
        pass

    def split_documents(self, docs: List[Document]) -> List[Document]:
        raise NotImplementedError


class SentenceTextSplitter(TextSplitter):
    def __init__(self, chunk_size):
        super().__init__()
        self.chunk_size = chunk_size

    """
    This Splitter will split our documents into sentences.
    """

    def split_documents(self, docs: List[Document]) -> List[Document]:
        split_docs = []
        for doc in docs:
            chunks = []
            total_chunk_len = 0
            sentences = doc.page_content.split(".")
            # not very effective split since it might mistake with "." in "Mr. Smith" or "Dr. Brown"
            for sentence in sentences:
                sentence = sentence.strip()
                if (sentence != "") and (len(sentence) > 10):
                    chunks.append(sentence)
                    total_chunk_len += len(sentence)
                if total_chunk_len > self.chunk_size:
                    split_docs.append(
                        Document(page_content=". ".join(chunks), metadata=doc.metadata)
                    )
                    total_chunk_len = 0
                    chunks = []
        return split_docs


class CharacterTextSplitter(TextSplitter):
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
        start_time = time.perf_counter()
        split_docs: List[Document] = corpus
        if self.is_split:
            # splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
            splitter = SentenceTextSplitter(chunk_size=200)
            split_docs = splitter.split_documents(corpus)

        # This might be to slow for 15000 split_docs
        for doc_id, doc in enumerate(split_docs):
            self.vector_db.append(
                {
                    "id": doc_id,
                    "text": doc.page_content,
                    "embedding": self.embed_text(doc.page_content),
                }
            )

        self.save_index()
        end_time = time.perf_counter()
        exec_time = end_time - start_time
        logger.log(f"Buiding index for retrieval took {exec_time:.2f} seconds")

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
        query_embedding = self.vector_db.embed_text(user_query)
        for item in self.vector_db:
            similarity = cosine_similarity(query_embedding, item["embedding"])
            similarities.append((item["text"], similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[: self.top_n]
