def chunk_text(text: str) -> list[str]:

    paragraphs = text.split("\n\n")

    chunks = []

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if paragraph:
            chunks.append(paragraph)

    return chunks
