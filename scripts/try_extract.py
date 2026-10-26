"""Manual check: print what ingestion extracts from a file."""
import sys

from ner_app.errors import NerError
from ner_app.ingestion import extract_document

try:
    doc = extract_document(sys.argv[1])
except NerError as exc:
    print(f"ERROR: {exc}")
    sys.exit(exc.exit_code)

print(f"format={doc.detected_format} chars={len(doc.text)} warnings={[w.code for w in doc.warnings]}")
print("-" * 40)
print(doc.text[:1500])
