#examples/loader_demo.py
from pathlib import Path
from metarag import DocumentLoader

DATA_DIR = Path(__file__).resolve().parent.parent / "tests" / "data"

loader = DocumentLoader(DATA_DIR)

report = loader.load()

print("=" * 60)
print("DocumentLoader API Demo")
print("=" * 60)

# ------------------------------------------------------------------
# DocumentList behaviour
# ------------------------------------------------------------------

print("\n=== DocumentList ===")

print(type(report))
print(f"Documents Extracted    : {len(report)}")

if len(report):
    print(f"First document         : {report[0].metadata['source']}")
    print(f"Last document          : {report[-1].metadata['source']}")

# ------------------------------------------------------------------
# Loaded summary
# ------------------------------------------------------------------

print("\n=== Loaded Summary ===")

print(f"Files loaded           : {report.loaded.count}")
print(f"Files          : {report.loaded.files}")

# ------------------------------------------------------------------
# Skipped summary
# ------------------------------------------------------------------

print("\n=== Skipped Summary ===")

print(f"Files skipped          : {report.skipped.count}")
print(f"Files          : {report.skipped.files}")

# ------------------------------------------------------------------
# Per-extension statistics
# ------------------------------------------------------------------

print("\n=== By Extension ===")

for ext, stats in report.loaded.by_extension.items():
    print(f"\nLoaded .{ext}")
    print(f"Count : {stats.count}")
    print(f"Files : {stats.files}")

for ext, stats in report.skipped.by_extension.items():
    print(f"\nSkipped .{ext}")
    print(f"Count  : {stats.count}")
    print(f"Files  : {stats.files}")
    print(f"Reason : {stats.reason}")

# ------------------------------------------------------------------
# Direct access
# ------------------------------------------------------------------

print("\n=== Direct Access ===")

if "pdf" in report.loaded.by_extension:
    print("PDF Count :", report.loaded["pdf"].count)
    print("PDF Files :", report.loaded["pdf"].files)

if "docx" in report.loaded.by_extension:
    print("DOCX Count :", report.loaded["docs"].count)
    print("DOCX Files :", report.loaded["docx"].files)

if "txt" in report.loaded.by_extension:
    print("TXT Count :", report.loaded["txt"].count)
    print("TXT Files :", report.loaded["txt"].files)

# ------------------------------------------------------------------
# Iterate over documents
# ------------------------------------------------------------------

print("\n=== Documents ===")

for i, doc in enumerate(report[:3], start=1):
    print(f"\nDocument {i}")
    print("Source :", doc.metadata.get("source"))
    print("Length :", len(doc.text))