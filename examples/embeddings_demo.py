from pathlib import Path

from metarag import CachedEmbeddings


# ============================================================
# Fake Embedding Model
# (Replace with SentenceTransformer/OpenAI later if desired)
# ============================================================

class DemoEmbeddingModel:

    model_name = "demo-embedding-model"

    def embed_query(self, text):

        print(f"Embedding query  : {text}")

        return [float(len(text)), 1.0, 2.0]

    def embed_documents(self, texts):

        print(f"Embedding {len(texts)} documents...")

        return [
            [float(len(t)), 1.0, 2.0]
            for t in texts
        ]


CACHE_DIR = Path(".metarag/demo_embedding_cache")

print("=" * 65)
print("MetaRAG Cached Embeddings Demo")
print("=" * 65)

model = DemoEmbeddingModel()

embedder = CachedEmbeddings(
    model=model,
    cache_dir=CACHE_DIR,
)

# ============================================================
# Single Query
# ============================================================

print("\n=== Single Query ===")

query = "What is Retrieval-Augmented Generation?"

vector = embedder.embed(query)

print(f"Embedding Dimension : {len(vector)}")
print(f"Embedding           : {vector}")

# ============================================================
# Cache Demonstration
# ============================================================

print("\n=== Cache Demonstration ===")

print("Embedding same query again...")

vector2 = embedder.embed(query)

print("Returned Vector")

print(vector2)

# ============================================================
# Batch Embedding
# ============================================================

print("\n=== Batch Embedding ===")

documents = [
    "MetaRAG is a modular RAG framework.",
    "Embeddings convert text into vectors.",
    "Vector databases perform similarity search.",
]

vectors = embedder.embed_documents(documents)

print(f"Documents Embedded : {len(vectors)}")

for i, vec in enumerate(vectors, start=1):

    print(f"Document {i} -> {vec}")

# ============================================================
# Cache Files
# ============================================================

print("\n=== Cache Files ===")

cache_files = list(CACHE_DIR.glob("*.npy"))

print(f"Cache Directory : {CACHE_DIR}")
print(f"Cached Files    : {len(cache_files)}")

for file in cache_files:

    print(file.name)

# ============================================================
# Persistence Demo
# ============================================================

print("\n=== Cache Persistence ===")

embedder2 = CachedEmbeddings(
    model=model,
    cache_dir=CACHE_DIR,
)

vector3 = embedder2.embed(query)

print("Loaded embedding using a new CachedEmbeddings instance.")

print(vector3)

print("\nDone.")