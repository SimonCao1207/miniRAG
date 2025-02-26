# https://huggingface.co/blog/ngxson/make-your-own-rag

import dspy

from miniRAG.retriever import Retriever
from miniRAG.utils.log import Logger

logger = Logger()


class QASignature(dspy.Signature):
    query = dspy.InputField(desc="A natural language question.")
    context = dspy.InputField(desc="A relevant context for answer the question.")
    answer = dspy.OutputField(desc="A concise factual answer.")


class RAG:
    def __init__(self, model, retriever: Retriever):
        self.model = model
        self.retriever = retriever

    def generate(self, query: str) -> str:
        logger.log_chat(query, "Query")
        retrieved_knowledge = self.retriever.retrieve(query)
        if retrieved_knowledge is None:
            return "No knowledge retrieved."

        dspy.configure(lm=self.model)

        logger.log_rule("Retrieved knowledge:")
        for chunk, similarity in retrieved_knowledge:
            print(f" - (similarity: {similarity:.2f}) {chunk}")

        module = dspy.ChainOfThought(QASignature)
        response = module(query=query, context=retrieved_knowledge)

        logger.log_chat(response.answer, "Chat response")

        return response.answer  # type ignore
