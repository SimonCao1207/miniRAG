import json
import os
import time
from pathlib import Path
from typing import List

import faiss
import numpy as np
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
    def __init__(self, index_path, embedding, is_split=False):
        self.index_path = Path(index_path)
        self.doc_path = Path(os.path.splitext(index_path)[0] + ".json")
        self.doc_map = []
        self.embedding = embedding
        self.is_split = is_split
        self.index = None

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
            splitter = SentenceTextSplitter(chunk_size=200)
            split_docs = splitter.split_documents(corpus)

        logger.log(f"Number of docs: {len(split_docs)}")

        embeddings = []

        # This might be to slow for 15000 split_docs
        for doc_id, doc in enumerate(split_docs):
            try:
                embeddings.append(self.embed_text(doc.page_content))
                self.doc_map.append(
                    {
                        "text": doc.page_content,
                        "metadata": doc.metadata,
                    }
                )
                end_time = time.perf_counter()
                exec_time = end_time - start_time
                start_time = end_time
                logger.log(
                    f"Embedding doc {doc_id}/{len(split_docs)} took {exec_time:.2f} seconds"
                )
            except Exception as e:
                logger.log(f"Error embedding doc {doc_id}: {e}", style="bold red")

                smaller_chunks = CharacterTextSplitter(
                    chunk_size=500, chunk_overlap=50
                ).split_documents([doc])

                logger.log(
                    f"Split the doc into {len(smaller_chunks)} smaller chunks and embed them"
                )
                for chunk in smaller_chunks:
                    try:
                        embeddings.append(self.embed_text(chunk.page_content))
                        self.doc_map.append(
                            {
                                "text": chunk.page_content,
                                "metadata": chunk.metadata,
                            }
                        )
                        end_time = time.perf_counter()
                        exec_time = end_time - start_time
                        start_time = end_time
                    except Exception as e:
                        logger.log(f"Error embedding chunk: {e}")

        embeddings = np.array(embeddings).astype("float32")
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)  # type: ignore

        self.save_index()
        end_time = time.perf_counter()
        exec_time = end_time - start_time
        logger.log(f"Buiding index for retrieval took {exec_time:.2f} seconds")

    def save_index(self):
        faiss.write_index(self.index, str(self.index_path))
        with open(self.doc_path.with_suffix(".json"), "w") as f:
            json.dump(self.doc_map, f)
            logger.log(f"Vector database built with {len(self.doc_map)} entries")

    def load(self):
        self.index = faiss.read_index(str(self.index_path))
        with open(self.doc_path.with_suffix(".json"), "r") as f:
            self.doc_map = json.load(f)
            logger.log(f"Vector database loaded with {len(self.doc_map)} entries")

    def __iter__(self):
        return iter(self.doc_map)

    def __len__(self):
        return len(self.doc_map)


class Retriever:
    def __init__(self, vector_db: VectorDB, top_n=3):
        self.vector_db = vector_db
        self.top_n = top_n

    def retrieve(self, user_query):
        """
        Given a user input, relevant splits are retrieved from storage using a Retriever.
        """

        query_embedding = np.array([self.vector_db.embed_text(user_query)]).astype(
            "float32"
        )
        if self.vector_db.index:
            distances, indices = self.vector_db.index.search(
                query_embedding, k=self.top_n
            )  # type: ignore
            results = [
                (self.vector_db.doc_map[idx]["text"], distances[0][i])
                for i, idx in enumerate(indices[0])
            ]
            return results
