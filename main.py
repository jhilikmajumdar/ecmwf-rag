from src.ingestion import load_document
from src.chunking import chunk_text
from src.embeddings import EmbeddingModel
from src.retriever import Retriever


# Load and chunk ECMWF documentation
document = load_document("data/ecmwf_notes.txt")
chunks = chunk_text(document)

# Create embeddings
embedding_model = EmbeddingModel()
chunk_embeddings = embedding_model.encode(chunks)

# Create retriever
retriever = Retriever(
    chunks,
    chunk_embeddings
)


# --------------------------------
# Small retrieval evaluation set
# --------------------------------

test_cases = [
    {
        "question": "What is ERA5?",
        "expected_text": "fifth-generation atmospheric reanalysis"
    },
    {
        "question": "What format are ERA5 model-level parameters stored in?",
        "expected_text": "GRIB2"
    },
    {
        "question": "How can Python retrieve ERA5 datasets?",
        "expected_text": "CDS API"
    },
    {
        "question": "What is ecCodes used for?",
        "expected_text": "decoding and encoding GRIB and BUFR"
    }
]


correct = 0


for case in test_cases:

    question = case["question"]
    expected_text = case["expected_text"]

    query_embedding = embedding_model.encode(
        [question]
    )[0]

    results = retriever.retrieve(
        query_embedding,
        top_k=1
    )

    retrieved_text = results[0]["text"]
    score = results[0]["score"]

    is_correct = expected_text.lower() in retrieved_text.lower()

    if is_correct:
        correct += 1

    print("\n================================")
    print("Question:")
    print(question)

    print("\nExpected information:")
    print(expected_text)

    print("\nRetrieved chunk:")
    print(retrieved_text)

    print(f"\nSimilarity score: {score:.3f}")

    print("\nRetrieval correct?")
    print(is_correct)


accuracy = correct / len(test_cases)

print("\n================================")
print("Retrieval accuracy:")
print(f"{accuracy:.2%}")
