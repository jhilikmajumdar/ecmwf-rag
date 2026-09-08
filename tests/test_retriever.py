import numpy as np

from src.retriever import Retriever


def test_retriever_returns_best_match():

    chunks = [
        "ERA5 overview",
        "CDS API retrieval",
        "ecCodes GRIB",
    ]

    embeddings = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
    ])

    retriever = Retriever(
        chunks,
        embeddings,
    )

    query_embedding = np.array(
        [0.0, 1.0]
    )

    results = retriever.retrieve(
        query_embedding,
        top_k=1,
    )

    assert results[0]["text"] == (
        "CDS API retrieval"
    )
