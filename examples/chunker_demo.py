from pathlib import Path

from metarag import DocumentLoader, Chunker

# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path(__file__).resolve().parent.parent / "tests" / "data"

CHUNK_SIZE = 500
OVERLAP = 50

STRATEGIES = [
    "fixed",
    "recursive",
    "semantic",
    "sentence",
    "sliding_window",
    "markdown",
]

# ============================================================
# Load Documents
# ============================================================

print("=" * 70)
print("MetaRAG Chunker API Demo")
print("=" * 70)

loader = DocumentLoader(DATA_DIR)
documents = loader.load(verbose=False)

print("\n=== Document Summary ===")
print(f"Files Loaded      : {documents.loaded.count}")
print(f"Documents Loaded  : {len(documents)}")
print(f"Files             : {documents.loaded.files}")

summary = []

# ============================================================
# Demonstrate Every Strategy
# ============================================================

for strategy in STRATEGIES:

    print("\n" + "=" * 70)
    print(f"Strategy : {strategy.upper()}")
    print("=" * 70)

    chunker = Chunker(
        strategy=strategy,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
    )

    chunks = chunker.chunk_documents(documents)

    summary.append((strategy, len(chunks)))

    print(f"Chunk Size        : {CHUNK_SIZE}")
    print(f"Overlap           : {OVERLAP}")
    print(f"Chunks Generated  : {len(chunks)}")

    if not chunks:
        print("No chunks generated.")
        continue

    first = chunks[0]

    print("\n--- First Chunk ---")

    print(f"Characters        : {len(first.text)}")
    print(f"Source            : {first.source}")
    print(f"Start Index       : {first.start_idx}")
    print(f"End Index         : {first.end_idx}")

    print("\nMetadata")

    if first.metadata:
        for key, value in first.metadata.items():
            print(f"  {key:<15}: {value}")
    else:
        print("  None")

    preview = first.text.replace("\n", " ")

    if len(preview) > 350:
        preview = preview[:350] + "..."

    print("\nPreview")
    print("-" * 70)
    print(preview)

# ============================================================
# Strategy Comparison
# ============================================================

print("\n")
print("=" * 70)
print("Chunking Strategy Summary")
print("=" * 70)

print(f"{'Strategy':<20}{'Chunks Generated'}")
print("-" * 40)

for strategy, count in summary:
    print(f"{strategy:<20}{count}")

# ============================================================
# Example Chunk Object
# ============================================================

print("\n")
print("=" * 70)
print("Example Chunk Object")
print("=" * 70)

example_chunker = Chunker(
    strategy="recursive",
    chunk_size=CHUNK_SIZE,
    overlap=OVERLAP,
)

example_chunks = example_chunker.chunk_documents(documents)

if example_chunks:

    chunk = example_chunks[0]

    print("Each Chunk stores:\n")

    print(f"text           : <{len(chunk.text)} characters>")
    print(f"source         : {chunk.source}")
    print(f"start_idx      : {chunk.start_idx}")
    print(f"end_idx        : {chunk.end_idx}")

    print("\nmetadata")

    if chunk.metadata:
        for key, value in chunk.metadata.items():
            print(f"  {key:<15}: {value}")

    print("\nAvailable attributes")
    print("------------------------------")
    print("chunk.text")
    print("chunk.source")
    print("chunk.start_idx")
    print("chunk.end_idx")
    print("chunk.metadata")

print("\nDone.")