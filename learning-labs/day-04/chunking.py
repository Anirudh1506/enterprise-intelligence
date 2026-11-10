from uuid import uuid4

# fixed size character chunking
def chunk_text(text: str, document_id:str, chunkSize: int = 500, overlap: int = 100) -> list[dict]:
    if not text or not text.strip():
        raise ValueError("Text must not be empty")
    if not isinstance(chunkSize, int) or chunkSize <= 0:
        raise ValueError("Chunk Size should be a positive non-zero integer")
    if not isinstance(overlap, int) or overlap < 0:
        raise ValueError("Overlap should not be negative integer")
    if overlap >= chunkSize:
        raise ValueError("Overlap must be smaller than the chunk size")

    result = []
    start = 0
    step = chunkSize - overlap

    while True:
        end = min(start + chunkSize, len(text))
        result.append({
            "chunk_id" : str(uuid4()),
            "document_id": document_id,
            "chunk_index": len(result)+1,
            "content": text[start:end],
            "start_offset": start,
            "end_offset": end
        })
        if end >= len(text):
            break
        start += step

    return result



if __name__ == "__main__":
    print(chunk_text("Hello All! I am Anirudh Koundinya Kasichainula", "test_id", 3, 1))
