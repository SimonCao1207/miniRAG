# https://huggingface.co/blog/ngxson/make-your-own-rag
from pathlib import Path

import datasets
from datasets import Dataset
from miniRAG.models import ChatModel
from miniRAG.retriever import Retriever
from miniRAG.utils.log import Logger

logger = Logger()


class RAG:
    def __init__(self, model: ChatModel, retriever: Retriever):
        self.model = model
        self.retriever = retriever

    def generate(self, query: str) -> str:
        retrieved_knowledge = self.retriever.retrieve(query)
        logger.log_rule("Retrieved knowledge:")
        for chunk, similarity in retrieved_knowledge:
            print(f" - (similarity: {similarity:.2f}) {chunk}")

        # TODO: make a better instruction prompt
        instruction_prompt = (
            """You are a helpful chatbot.\nUse only the following context:\n"""
            + "\n".join([f" - {chunk}" for chunk, _ in retrieved_knowledge])
        )
        logger.log_task(
            instruction_prompt, "Instruction Prompt", subtitle=self.model.model_id
        )
        messages = [
            {"role": "system", "content": instruction_prompt},
            {"role": "user", "content": query},
        ]
        response = self.model(messages)
        logger.log_rule("Chatbot response:")

        return response.content  # type ignore


def load_corpus(corpus_path: Path) -> Dataset:
    """
    Return Dataset object with features "id" and "contents"
    """
    logger.log("Loading the corpus...")
    if corpus_path.suffix == ".jsonl":
        corpus = datasets.load_dataset(  # type: ignore
            "json", data_files=str(corpus_path), split="train"
        )
        return corpus
    else:
        with open(corpus_path, "r") as file:
            data = [line.strip() for line in file.readlines()]
        dataset = Dataset.from_dict(
            {
                "id": list(range(len(data))),
                "contents": data,
            }
        )
        return dataset
