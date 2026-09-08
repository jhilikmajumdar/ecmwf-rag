import numpy as np


class Retriever:

    def __init__(self, chunks, embeddings):
        self.chunks = chunks
        self.embeddings = embeddings


    def retrieve(self, query_embedding, top_k=2):

        scores = np.dot(
            self.embeddings,
            query_embedding
        )

        best_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in best_indices:

            results.append({
                "text": self.chunks[index],
                "score": float(scores[index])
            })

        return results
