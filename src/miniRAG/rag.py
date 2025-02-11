# https://huggingface.co/blog/ngxson/make-your-own-rag
import json
from pathlib import Path

import ollama
from log import Logger

logger = Logger()

EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
LANGUAGE_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"
DATASET_PATH = (
    Path(__file__).resolve().parent.parent.parent / "datasets" / "cat-facts.txt"
)
VECTOR_DB_PATH = Path(__file__).resolve().parent.parent / "vector_db.json"


def load_dataset():
    logger.log("Loading the datasets")
    with open(DATASET_PATH, "r") as file:
        return [line.strip() for line in file.readlines()]


def embed_text(text):
    return ollama.embed(model=EMBEDDING_MODEL, input=text)["embeddings"][0]


def build_vector_db(dataset):
    # Each element in the VECTOR_DB will be a tuple (chunk, embedding)
    # The embedding is a list of floats, for example: [0.1, 0.04, -0.34, 0.21, ...]
    # TODO: Use a more efficient vector database
    vector_db = [(chunk, embed_text(chunk)) for chunk in dataset]
    with open(VECTOR_DB_PATH, "w") as f:
        json.dump(vector_db, f)
    logger.log(f"Vector database built with {len(vector_db)} entries")


def load_vector_db():
    if VECTOR_DB_PATH.exists():
        with open(VECTOR_DB_PATH, "r") as f:
            vector_db = json.load(f)
            logger.log(f"Vector database loaded with {len(vector_db)} entries")
            return vector_db
    return []


def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x**2 for x in a]) ** 0.5
    norm_b = sum([x**2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)


def retrieve(query, vector_db, top_n=3):
    similarities = []
    for chunk, embedding in vector_db:
        similarity = cosine_similarity(embed_text(query), embedding)
        similarities.append((chunk, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]


def chat(query, retrieved_knowledge):
    # TODO: make a better instruction prompt
    instruction_prompt = (
        """You are a helpful chatbot.\nUse only the following context:\n"""
        + "\n".join([f" - {chunk}" for chunk, _ in retrieved_knowledge])
    )
    logger.log_task(instruction_prompt, "Instruction Prompt")
    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {"role": "system", "content": instruction_prompt},
            {"role": "user", "content": query},
        ],
        stream=True,
    )
    logger.log_rule("Chatbot response:")
    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)
    print()


if __name__ == "__main__":
    dataset = load_dataset()
    if not VECTOR_DB_PATH.exists():
        build_vector_db(dataset)
    vector_db = load_vector_db()

    # Chatbot
    input_query = input("Ask me a question: ")
    retrieved_knowledge = retrieve(input_query, vector_db)

    logger.log_rule("Retrieved knowledge:")
    for chunk, similarity in retrieved_knowledge:
        print(f" - (similarity: {similarity:.2f}) {chunk}")

    chat(input_query, retrieved_knowledge)
