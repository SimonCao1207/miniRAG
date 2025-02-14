# https://huggingface.co/blog/ngxson/make-your-own-rag
import argparse
import os
from pathlib import Path

from miniRAG.models import ChatModel, OllamaModel, OpenAIServerModel
from miniRAG.retriever import Retriever, VectorDB
from utils.log import Logger

logger = Logger()


class Config:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Configure RAG settings.")
        parser.add_argument(
            "--model", type=str, default="llama", help="Language model to use"
        )
        parser.add_argument(
            "--embedding_model",
            type=str,
            default="hf.co/CompendiumLabs/bge-base-en-v1.5-gguf",
            help="Embedding model to use",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            default=str(
                Path(__file__).resolve().parent.parent.parent
                / "datasets"
                / "cat-facts.txt"
            ),
            help="Path to dataset",
        )
        parser.add_argument(
            "--vector_db",
            type=str,
            default=str(Path(__file__).resolve().parent.parent / "vector_db.json"),
            help="Path to vector database",
        )
        args = parser.parse_args()

        self.EMBEDDING_MODEL = args.embedding_model
        self.LANGUAGE_MODEL = args.model
        self.DATASET_PATH = Path(args.dataset)
        self.VECTOR_DB_PATH = Path(args.vector_db)
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


_shortcut = {
    "gpt": "gpt-4o-mini",
    "llama": "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF",
}


class RAG:
    def __init__(self, model: ChatModel, retriever: Retriever):
        self.model = model
        self.retriever = retriever

    def generate(self, query):
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
            instruction_prompt, "Instruction Prompt", subtitle=model.model_id
        )
        messages = [
            {"role": "system", "content": instruction_prompt},
            {"role": "user", "content": query},
        ]
        response = self.model(messages)
        logger.log_rule("Chatbot response:")
        # for chunk in stream:
        #     print(chunk["message"]["content"], end="", flush=True)

        print(response.content)  # type: ignore


def load_dataset(path):
    logger.log("Loading the datasets")
    with open(path, "r") as file:
        return [line.strip() for line in file.readlines()]


if __name__ == "__main__":
    config = Config()
    dataset = load_dataset(config.DATASET_PATH)
    vector_db = VectorDB(config.VECTOR_DB_PATH, config.EMBEDDING_MODEL)
    if not config.VECTOR_DB_PATH.exists():
        vector_db.initialize(dataset)
    vector_db.load()
    retriever = Retriever(vector_db=vector_db)
    model_id = _shortcut[config.LANGUAGE_MODEL]
    if config.LANGUAGE_MODEL == "gpt":
        model = OpenAIServerModel(
            model_id=model_id,
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base="https://api.openai.com/v1",
        )
    elif config.LANGUAGE_MODEL == "llama":
        model = OllamaModel(model_id)
    rag = RAG(model, retriever)

    # Chatbot
    input_query = input("Ask me a question: ")
    rag.generate(input_query)
