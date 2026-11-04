"""Prepare NuNER for GLiNER fine-tuning (v2).

Fixes over v1:
- Tokenizes with the same regex as GLiNER's default word splitter (verified at startup).
- Matches entities as whole token sequences, never as substrings inside words.
- Labels EVERY occurrence of an entity string (v1 labeled only the first).
- Entity strings given conflicting labels are not guessed at.
- 'strict' policy (default) drops examples with any annotation we cannot trust
  (unmapped label, unaligned entity, label conflict), because GLiNER treats
  unlabeled spans as negatives and incomplete gold teaches it to under-predict.
- Counts every skip reason, keeps a few examples of each, reports unmapped labels.
- Writes *.eval.jsonl (original text + character offsets) for fair evaluation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

from datasets import load_dataset

# Same pattern as GLiNER's default word splitter (checked by verify_tokenizer below).
TOKEN_RE = re.compile(r"\w+(?:[-_]\w+)*|\S")

SPLITS = ("train", "validation", "test")
EXAMPLES_PER_REASON = 3

# Unchanged from v1 on purpose, so v1 and v2 differ only in how data is processed.
LABEL_MAP = {
    "person": "person", "person name": "person", "people": "person", "athlete": "person",
    "organization": "organization", "company": "organization",
    "company/organization": "organization", "business entity": "organization",
    "educational institution": "organization", "sports team": "organization",
    "business": "organization",
    "location": "location", "place": "location", "city": "location", "country": "location",
    "state": "location", "geographical location": "location", "geographic location": "location",
    "date": "date", "year": "date", "time period": "date", "date/time": "date",
    "time": "time",
    "event": "event",
    "product": "product", "product name": "product", "brand": "product",
    "software": "product", "device": "product",
    "technology": "technology", "operating system": "technology",
    "payment": "money", "currency": "money",
    "measurement": "quantity", "quantity": "quantity",
}


def get_split(text: str) -> str:
    """Identical to v1: the same text always lands in the same split."""
    bucket = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big") % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


def tokenize(text: str):
    matches = list(TOKEN_RE.finditer(text))
    return [m.group() for m in matches], [(m.start(), m.end()) for m in matches]


def find_all(tokens, target):
    """All (start, end_inclusive) token positions where `target` occurs."""
    n, first, hits = len(target), target[0], []
    for i in range(len(tokens) - n + 1):
        if tokens[i] == first and tuple(tokens[i : i + n]) == target:
            hits.append((i, i + n - 1))
    return hits


def verify_tokenizer(texts) -> str:
    try:
        from gliner.data_processing.tokenizer import WordsSplitter

        splitter = WordsSplitter("whitespace")
        mismatches = 0
        for text in texts:
            theirs = [(t, s, e) for t, s, e in splitter(text)]
            mine = [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]
            mismatches += theirs != mine
        return f"{mismatches} mismatches in {len(texts)} texts"
    except Exception as exc:  # noqa: BLE001 - API may differ between versions
        return f"NOT VERIFIED ({type(exc).__name__}: {exc})"


def convert_example(example, strict, max_tokens, unmapped_labels, events):
    """Return ((train_record, eval_record), None) or (None, skip_reason)."""
    text = example.get("input")
    if not isinstance(text, str) or not text.strip():
        return None, "bad_text"

    raw_output = example.get("output")
    if isinstance(raw_output, list):
        annotations = raw_output
    else:
        try:
            annotations = ast.literal_eval(raw_output)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            return None, "bad_output"
    if not isinstance(annotations, list):
        return None, "bad_output"

    tokens, offsets = tokenize(text)
    if len(tokens) < 3:
        return None, "too_few_tokens"
    if len(tokens) > max_tokens:
        return None, "too_long"

    mapped, has_unmapped, parsed_any = [], False, False
    for annotation in annotations:
        if not isinstance(annotation, str) or " <> " not in annotation:
            continue
        entity_text, raw_label = annotation.rsplit(" <> ", 1)
        entity_text, raw_label = entity_text.strip(), raw_label.strip()
        if not entity_text or not raw_label:
            continue
        parsed_any = True
        label = LABEL_MAP.get(raw_label.casefold())
        if label is None:
            has_unmapped = True
            unmapped_labels[raw_label.casefold()] += 1
        else:
            mapped.append((entity_text, label))

    if not parsed_any:
        return None, "no_parsable_annotations"
    if not mapped:
        return None, "no_mapped_labels"

    labels_by_entity = defaultdict(set)
    unaligned = 0
    for entity_text, label in mapped:
        entity_tokens = tuple(TOKEN_RE.findall(entity_text))
        if entity_tokens:
            labels_by_entity[entity_tokens].add(label)
        else:
            unaligned += 1

    spans, conflicts, folded = [], 0, None
    for entity_tokens, labels in labels_by_entity.items():
        if len(labels) > 1:
            conflicts += 1
            continue
        label = next(iter(labels))
        hits = find_all(tokens, entity_tokens)
        if not hits:
            if folded is None:
                folded = [t.casefold() for t in tokens]
            hits = find_all(folded, tuple(t.casefold() for t in entity_tokens))
            if hits:
                events["matched_ignoring_case"] += 1
        if not hits:
            unaligned += 1
            continue
        if len(hits) > 1:
            events["entity_with_multiple_occurrences"] += 1
        spans.extend((s, e, label) for s, e in hits)

    if conflicts and strict:
        return None, "label_conflict"
    if unaligned and strict:
        return None, "unaligned_entity"
    if not spans:
        return None, "no_aligned_entities"

    # Flat NER: keep the longest span when spans overlap.
    spans.sort(key=lambda s: (-(s[1] - s[0]), s[0]))
    occupied, kept = set(), []
    for s, e, label in spans:
        if occupied.intersection(range(s, e + 1)):
            events["overlapping_span_dropped"] += 1
            continue
        occupied.update(range(s, e + 1))
        kept.append((s, e, label))
    kept.sort()

    if strict and has_unmapped:
        events["dropped_only_because_of_unmapped_labels"] += 1
        return None, "has_unmapped_annotation"

    train_record = {"tokenized_text": tokens, "ner": [[s, e, l] for s, e, l in kept]}
    eval_record = {
        "text": text,
        "entities": [
            {"start": offsets[s][0], "end": offsets[e][1], "label": l} for s, e, l in kept
        ],
    }
    return (train_record, eval_record), None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="data/nuner_v2")
    parser.add_argument("--policy", choices=["strict", "lenient"], default="strict")
    parser.add_argument("--max-tokens", type=int, default=250)
    parser.add_argument("--limit", type=int, default=None, help="process only the first N rows")
    args = parser.parse_args()
    strict = args.policy == "strict"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading NuNER entity split...")
    dataset = load_dataset("numind/NuNER", split="entity")
    total = len(dataset) if args.limit is None else min(len(dataset), args.limit)

    tokenizer_check = verify_tokenizer([dataset[i]["input"] for i in range(min(2000, total))])
    print(f"Tokenizer check against GLiNER's splitter: {tokenizer_check}")
    if not tokenizer_check.startswith("0 mismatches"):
        print("  !! WARNING: tokenization may not match GLiNER. Tell Claude before training.")

    seen = set()
    skip_reasons, events, unmapped = Counter(), Counter(), Counter()
    label_counts, split_counts = Counter(), Counter()
    split_labels = {s: Counter() for s in SPLITS}
    skip_examples = defaultdict(list)

    def note_skip(reason, example):
        skip_reasons[reason] += 1
        if len(skip_examples[reason]) < EXAMPLES_PER_REASON:
            skip_examples[reason].append(
                {"input": str(example.get("input"))[:200], "output": str(example.get("output"))[:300]}
            )

    with ExitStack() as stack:
        def open_out(name):
            return stack.enter_context((out / name).open("w", encoding="utf-8", newline="\n"))

        train_files = {s: open_out(f"{s}.jsonl") for s in SPLITS}
        eval_files = {s: open_out(f"{s}.eval.jsonl") for s in ("validation", "test")}

        for index, example in enumerate(dataset):
            if index >= total:
                break
            text = example.get("input")
            if isinstance(text, str):
                key = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
                if key in seen:
                    note_skip("duplicate_text", example)
                    continue
                seen.add(key)

            result, reason = convert_example(example, strict, args.max_tokens, unmapped, events)
            if result is None:
                note_skip(reason, example)
                continue

            train_record, eval_record = result
            split = get_split(text)
            train_files[split].write(json.dumps(train_record, ensure_ascii=False) + "\n")
            if split in eval_files:
                eval_files[split].write(json.dumps(eval_record, ensure_ascii=False) + "\n")
            split_counts[split] += 1
            labels = [e[2] for e in train_record["ner"]]
            label_counts.update(labels)
            split_labels[split].update(labels)

            if (index + 1) % 100_000 == 0:
                print(f"Processed {index + 1:,} / {total:,} ...")

    kept = sum(split_counts.values())
    skipped = sum(skip_reasons.values())
    print(f"\n=== Done (policy={args.policy}, max_tokens={args.max_tokens}) ===")
    print(f"Processed {total:,} | kept {kept:,} | skipped {skipped:,}")
    for split in SPLITS:
        print(f"  {split}: {split_counts[split]:,}")

    print("\nSkip reasons (first failing check wins):")
    for reason, n in skip_reasons.most_common():
        print(f"  {reason:<28}{n:>10,}  ({n / total:.1%})")

    print("\nOther events:")
    for name, n in events.most_common():
        print(f"  {name:<44}{n:>10,}")

    print("\nExamples per skip reason:")
    for reason, items in skip_examples.items():
        print(f"  [{reason}]")
        for item in items:
            print(f"    input : {item['input']!r}")
            print(f"    output: {item['output']!r}")

    print("\nTop 40 UNMAPPED raw labels (candidates for LABEL_MAP review):")
    for label, n in unmapped.most_common(40):
        print(f"  {label:<40}{n:>9,}")

    print("\nKept entity distribution:")
    for label, n in label_counts.most_common():
        print(f"  {label:<14}{n:>9,}")
    if kept:
        print(f"  entities per kept example: {sum(label_counts.values()) / kept:.2f}")

    with (out / "unmapped_labels.tsv").open("w", encoding="utf-8", newline="\n") as f:
        for label, n in unmapped.most_common(500):
            f.write(f"{label}\t{n}\n")
    with (out / "stats.json").open("w", encoding="utf-8", newline="\n") as f:
        json.dump(
            {
                "policy": args.policy, "max_tokens": args.max_tokens, "processed": total,
                "kept": dict(split_counts), "skip_reasons": dict(skip_reasons),
                "events": dict(events), "label_counts": dict(label_counts),
                "split_label_counts": {s: dict(c) for s, c in split_labels.items()},
                "tokenizer_check": tokenizer_check,
            },
            f, indent=2,
        )
    print(f"\nWrote files to {out.resolve()}")


if __name__ == "__main__":
    main()