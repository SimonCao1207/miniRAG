# https://huggingface.co/blog/ngxson/make-your-own-rag

from miniRAG.models import ChatModel
from miniRAG.retriever import Retriever
from miniRAG.utils.log import Logger

logger = Logger()


class RAG:
    def __init__(self, model: ChatModel, retriever: Retriever):
        self.model = model
        self.retriever = retriever

    def generate(self, query: str) -> str:
        logger.log_chat(query, "Query")
        retrieved_knowledge = self.retriever.retrieve(query)
        if retrieved_knowledge is None:
            return "No knowledge retrieved."
        logger.log_rule("Retrieved knowledge:")
        for chunk, similarity in retrieved_knowledge:
            print(f" - (similarity: {similarity:.2f}) {chunk}")

        # TODO: make a better instruction prompt
        instruction_prompt = (
            """You are a helpful chatbot.\nUse only the following context:\n"""
            + "\n".join([f" - {chunk}" for chunk, _ in retrieved_knowledge])
        )
        logger.log_chat(
            instruction_prompt, "Instruction Prompt", subtitle=self.model.model_id
        )
        messages = [
            {"role": "system", "content": instruction_prompt},
            {"role": "user", "content": query},
        ]
        response = self.model(messages)
        logger.log_chat(response.content, "Chat response")

        return response.content  # type ignore
