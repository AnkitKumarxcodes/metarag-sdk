# examples/vector_db_demo.py
"""
VectorDB API demo — builds a real index from your test PDFs using Ollama
embeddings, then exercises InMemoryVectorDB's build/search/add.
"""

from pathlib import Path
import requests

from metarag import DocumentLoader, Chunker, CachedEmbeddings, InMemoryVectorDB

DATA_DIR = Path(__file__).resolve().parent.parent / "tests" / "data"


import concurrent.futures

class OllamaEmbeddings:
    """
    Minimal EmbeddingInterface implementation over Ollama's /api/embeddings.
    embed_documents() fires requests concurrently (I/O-bound — waiting on
    Ollama's HTTP response, not CPU-bound), instead of one-at-a-time.
    CachedEmbeddings only ever passes UNCACHED texts here, so this only
    runs at all for chunks not already in .metarag/embeddings/.
    """

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434", max_workers: int = 8):
        self.model = model
        self.base_url = base_url
        self.max_workers = max_workers

    def embed_query(self, text: str):
        resp = requests.post(f"{self.base_url}/api/embeddings", json={"model": self.model, "prompt": text})
        resp.raise_for_status()
        return resp.json()["embedding"]

    def embed_documents(self, texts):
        if not texts:
            return []

        results = [None] * len(texts)
        completed = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {executor.submit(self.embed_query, t): i for i, t in enumerate(texts)}
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
                completed += 1
                if completed % 10 == 0 or completed == len(texts):
                    print(f"  [Embeddings] {completed}/{len(texts)} chunks embedded...")

        return results


print("=" * 60)
print("VectorDB API Demo")
print("=" * 60)

# ------------------------------------------------------------------
# Build real chunks + embeddings from the test corpus
# ------------------------------------------------------------------

loader = DocumentLoader(DATA_DIR)
docs = loader.load(verbose=False)
print(f"\nDocuments loaded : {len(docs)}")

chunker = Chunker(strategy="recursive", chunk_size=500, overlap=50)
chunks = chunker.chunk_documents(docs)
print(f"Chunks produced  : {len(chunks)}")

embeddings = CachedEmbeddings(OllamaEmbeddings())
chunk_texts = [c.text for c in chunks]
chunk_embeddings = embeddings.embed_documents(chunk_texts)
print(f"Embeddings ready : {len(chunk_embeddings)} vectors, dim={len(chunk_embeddings[0])}")

# ------------------------------------------------------------------
# build()
# ------------------------------------------------------------------

print("\n=== build() ===")

vdb = InMemoryVectorDB()
vdb.build(chunks, chunk_embeddings)
print(f"Chunks stored : {len(vdb.chunks)}")

# ------------------------------------------------------------------
# search()
# ------------------------------------------------------------------

print("\n=== search() ===")

query = "What is the main topic of this document?"
query_embedding = embeddings.embed_query(query)
results = vdb.search(query_embedding, k=3)

print(f"Query : '{query}'")
for i, (chunk, score) in enumerate(results, start=1):
    text = chunk.text if hasattr(chunk, "text") else str(chunk)
    print(f"\n  #{i}  score={score:.4f}")
    print(f"      {text[:100]}...")

# ------------------------------------------------------------------
# add() — append a synthetic chunk, confirm it's searchable
# ------------------------------------------------------------------

print("\n=== add() ===")

from metarag import Chunk
new_chunk = Chunk(text="MetaRAG unit test marker chunk.", metadata={"source": "synthetic"})
new_embedding = embeddings.embed_query(new_chunk.text)

before = len(vdb.chunks)
vdb.add([new_chunk], [new_embedding])
after = len(vdb.chunks)
print(f"Chunks before add : {before}")
print(f"Chunks after add  : {after}")

marker_results = vdb.search(embeddings.embed_query("unit test marker chunk"), k=1)
top_text = marker_results[0][0].text if hasattr(marker_results[0][0], "text") else str(marker_results[0][0])
print(f"Top result after add : {top_text[:60]}")

# ------------------------------------------------------------------
# save() — InMemoryVectorDB doesn't persist, just confirms the warning
# ------------------------------------------------------------------

print("\n=== save() ===")
vdb.save()