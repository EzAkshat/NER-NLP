"""Compare model output with a hand-made gold file.

Usage:
  python scripts/evaluate.py DOC GOLD.json          # score
  python scripts/evaluate.py DOC GOLD.json --draft   # write model output as a draft gold file to correct
"""

import json
import sys
from collections import Counter
from pathlib import Path

from ner_app.pipeline import analyze_file


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main(argv):
    doc, gold_path = argv[0], Path(argv[1])
    result = analyze_file(doc)
    pred = Counter((m.label, m.text) for m in result.mentions)

    if "--draft" in argv:
        draft = [{"label": l, "text": t} for (l, t), n in pred.items() for _ in range(n)]
        gold_path.write_text(json.dumps(draft, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {len(draft)} draft entries to {gold_path}. Correct it by hand, then rerun without --draft.")
        return

    gold = Counter(
        (g["label"], g["text"]) for g in json.loads(gold_path.read_text(encoding="utf-8"))
    )
    tp = pred & gold
    print(f"{'label':<12}{'prec':>7}{'rec':>7}{'f1':>7}   tp/fp/fn")
    for label in sorted({l for l, _ in pred} | {l for l, _ in gold}) + ["ALL"]:
        keep = (lambda k: True) if label == "ALL" else (lambda k, lab=label: k[0] == lab)
        t = sum(n for k, n in tp.items() if keep(k))
        fp = sum(n for k, n in (pred - gold).items() if keep(k))
        fn = sum(n for k, n in (gold - pred).items() if keep(k))
        p, r, f = prf(t, fp, fn)
        print(f"{label:<12}{p:>7.2f}{r:>7.2f}{f:>7.2f}   {t}/{fp}/{fn}")

    print("\nSpurious (model said, gold doesn't):")
    for (l, t), n in sorted((pred - gold).items()):
        print(f"  {l:<10} {t!r} x{n}")
    print("\nMissed (gold has, model didn't):")
    for (l, t), n in sorted((gold - pred).items()):
        print(f"  {l:<10} {t!r} x{n}")


if __name__ == "__main__":
    main(sys.argv[1:])