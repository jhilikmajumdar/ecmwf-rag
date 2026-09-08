import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)


class RAGPipeline:

    def __init__(self, retriever, embedding_model):

        self.retriever = retriever
        self.embedding_model = embedding_model

        model_name = "google/flan-t5-small"

        # Converts text into tokens understood by FLAN-T5
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        # Load the actual language model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name
        )

        self.model.eval()


    def answer(self, question: str) -> str:

        # --------------------------------
        # R: RETRIEVAL
        # --------------------------------

        query_embedding = self.embedding_model.encode(
            [question]
        )[0]

        results = self.retriever.retrieve(
            query_embedding,
            top_k=1
        )

        context = results[0]["text"]


        # --------------------------------
        # A: AUGMENTATION
        # --------------------------------

        prompt = f"""
Answer the question using only the context below.

If the answer is not in the context,
say: I do not know based on the provided context.

Context:
{context}

Question:
{question}

Answer:
"""


        # --------------------------------
        # TOKENIZATION
        # --------------------------------

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )


        # --------------------------------
        # G: GENERATION
        # --------------------------------

        with torch.inference_mode():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False
            )


        # Convert generated tokens back to text
        answer = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return answer
