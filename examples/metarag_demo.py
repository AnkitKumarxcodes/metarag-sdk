# examples/metarag_demo.py
"""Full MetaRAG walkthrough — fit, ask, benchmark, and every observability output."""

from pathlib import Path
import requests
from metarag import MetaRAG, OllamaGenerator

DATA_DIR = Path(__file__).resolve().parent.parent / "tests" / "data"


import concurrent.futures
from metarag import CachedEmbeddings


class OllamaEmbeddings:
    def __init__(self, model="nomic-embed-text", base_url="http://localhost:11434", max_workers=8):
        self.model, self.base_url, self.max_workers = model, base_url, max_workers

    def embed_query(self, text):
        r = requests.post(f"{self.base_url}/api/embeddings", json={"model": self.model, "prompt": text})
        r.raise_for_status()
        return r.json()["embedding"]

    def embed_documents(self, texts):
        if not texts:
            return []
        results = [None] * len(texts)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(self.embed_query, t): i for i, t in enumerate(texts)}
            done = 0
            for f in concurrent.futures.as_completed(futures):
                results[futures[f]] = f.result()
                done += 1
                if done % 10 == 0 or done == len(texts):
                    print(f"  [Embeddings] {done}/{len(texts)}")
        return results
    
embeddings = CachedEmbeddings(OllamaEmbeddings())

rag = MetaRAG(
    docs=str(DATA_DIR),
    embeddings=embeddings,
    generator=OllamaGenerator(model="mistral"),
    project="metarag_demo",
    k=3,
)

print("\n### fit() ###")
rag.fit()

print("\n### ask() ###")
answer = rag.ask("What is the main topic of this document?")
print(answer)

QUERIES = [
    "What is the main topic of this document?",
    "Summarize the key points.",
    "What numbers or figures are mentioned?",
]

print("\n### benchmark() ###")
df = rag.benchmark(QUERIES, retrieval_only=True, train_router=True, save_csv=True)
print(f"Benchmark rows: {len(df)}  ->  {rag._benchmark_path}")

print("\n### status() ###")
rag.status()

print("\n### leaderboard() ###")
rag.leaderboard()

print("\n### dashboard() ###")
rag.dashboard()

print("\n### report() ###")
rag.report()

print("\n### pipeline_graph('hybrid') ###")
rag.pipeline_graph("hybrid")

print("\n### inspect() ###")
rag.inspect(QUERIES[0], k=2)

print("\n### trace() ###")
rag.trace(QUERIES[0], pipeline_name="full")

print("\n### explain() ###")
print(rag.explain(QUERIES[0]))

print("\n### analyze_query() / analyze_corpus() ###")
print(rag.analyze_query(QUERIES[0]))
print(rag.analyze_corpus())

print("\n### save() ###")
rag.save()

print(f"\nDone. Files at: {rag._base}")