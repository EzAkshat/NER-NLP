"""Compare models on the hand-made real-document suite (exact label + text matches).

Gold and Flair predictions are mapped into the 10-label taxonomy the fine-tuned models use,
so every model is scored on the same labels. CARDINAL and ORDINAL gold entities have no
equivalent label and are left out for every model.

Run: .venv-train/Scripts/python.exe scripts/suite_compare.py [--strip-titles] [--ensemble A1000]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import evaluate_v2 as ev

GOLD_MAP = {
    "PERSON": "person", "ORG": "organization",
    "GPE": "location", "LOC": "location", "FAC": "location",
    "DATE": "date", "TIME": "time", "EVENT": "event", "PRODUCT": "product",
    "MONEY": "money", "PERCENT": "quantity", "QUANTITY": "quantity",
}
TOKEN_RE = re.compile(r"\w+(?:[-_]\w+)*|\S")
TITLE_RE = re.compile(
    r"^(?:(?:dr|prof|mr|mrs|ms|mx|sir|shri|smt|sri|mme|mlle|rev|capt|col|gen|lt|sgt)\.?\s+)+",
    re.IGNORECASE,
)
MAX_CHUNK_TOKENS = 200
DEFAULT_MODELS = [
    "base=urchade/gliner_medium-v2.1",
    "A1000=models/gliner-v2/rawA/checkpoint-1000",
]


def read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    try:
        sys.path.insert(0, "src")
        from ner_app.ingestion.normalize import normalize_text
        return normalize_text(text)
    except Exception:  # noqa: BLE001 - fall back to the raw text
        return text


def split_text(text: str):
    """Pieces of at most MAX_CHUNK_TOKENS tokens, cut at line breaks where possible."""
    pieces, cur = [], None  # cur = [start, end, token_count]

    def flush():
        nonlocal cur
        if cur:
            pieces.append((cur[0], cur[1]))
            cur = None

    for m in re.finditer(r"[^\n]*\n?", text):
        ls, le = m.span()
        if le == ls:
            continue
        toks = list(TOKEN_RE.finditer(text[ls:le]))
        if len(toks) > MAX_CHUNK_TOKENS:  # one very long line: cut at token boundaries
            flush()
            for i in range(0, len(toks), MAX_CHUNK_TOKENS):
                group = toks[i:i + MAX_CHUNK_TOKENS]
                pieces.append((ls + group[0].start(), ls + group[-1].end()))
            continue
        if cur and cur[2] + len(toks) > MAX_CHUNK_TOKENS:
            flush()
        if cur is None:
            cur = [ls, le, len(toks)]
        else:
            cur[1], cur[2] = le, cur[2] + len(toks)
    flush()
    return pieces


def load_gold(path: Path):
    gold, dropped = Counter(), Counter()
    for g in json.loads(path.read_text(encoding="utf-8")):
        label = GOLD_MAP.get(g["label"])
        if label is None:
            dropped[g["label"]] += 1
        else:
            gold[(label, g["text"])] += 1
    return gold, dropped


def to_counters(chunk_preds, texts, chunk_meta, threshold, strip_titles=False):
    per_doc = [Counter() for _ in texts]
    for plist, (d, offset) in zip(chunk_preds, chunk_meta):
        for s, e, label, score in plist:
            if score < threshold:
                continue
            text = texts[d][s + offset:e + offset]
            if strip_titles and label == "person":
                text = TITLE_RE.sub("", text) or text
            per_doc[d][(label, text)] += 1
    return per_doc


def tally(preds, golds, label=None):
    tp = fp = fn = 0
    for p, g in zip(preds, golds):
        if label:
            p = Counter({k: v for k, v in p.items() if k[0] == label})
            g = Counter({k: v for k, v in g.items() if k[0] == label})
        tp += sum((p & g).values())
        fp += sum((p - g).values())
        fn += sum((g - p).values())
    return tp, fp, fn


def lenient_tally(preds, golds):
    """Boundary-tolerant: a prediction matches a gold entity of the same label when one text
    contains the other (e.g. 'Dr. Priya Desai' vs 'Priya Desai'). Each prediction is used once."""
    tp = fp = fn = 0
    for p, g in zip(preds, golds):
        exact = p & g
        tp += sum(exact.values())
        leftover = list((p - exact).elements())
        missed = 0
        for (label, gold_text), n in (g - exact).items():
            for _ in range(n):
                hit = next((i for i, (pl, pt) in enumerate(leftover)
                            if pl == label and (gold_text in pt or pt in gold_text)), None)
                if hit is None:
                    missed += 1
                else:
                    leftover.pop(hit)
                    tp += 1
        fp += len(leftover)
        fn += missed
    return tp, fp, fn


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="samples/real/suite")
    ap.add_argument("--model", action="append", default=[], help="name=path (repeatable)")
    ap.add_argument("--thresholds", default="0.5,0.6,0.7")
    ap.add_argument("--detail-threshold", type=float, default=0.7)
    ap.add_argument("--baseline", default="base")
    ap.add_argument("--ensemble", default=None, help="model name to merge with Flair")
    ap.add_argument("--misses", default=None, help="model name to list misses/spurious for")
    ap.add_argument("--strip-titles", action="store_true",
                    help="drop Dr./Prof./Mr. ... from the start of person text")
    ap.add_argument("--no-flair", action="store_true")
    ap.add_argument("--flair-cpu", action="store_true")
    args = ap.parse_args()

    specs = args.model or DEFAULT_MODELS
    thresholds = sorted({float(x) for x in args.thresholds.split(",")} | {args.detail_threshold})
    docs = sorted(p for p in Path(args.dir).glob("*.txt") if p.with_suffix(".gold.json").exists())
    if not docs:
        raise SystemExit(f"No NAME.txt + NAME.gold.json pairs found in {args.dir}")

    texts, golds, dropped = [], [], Counter()
    for p in docs:
        texts.append(read_text(p))
        g, d = load_gold(p.with_suffix(".gold.json"))
        golds.append(g)
        dropped.update(d)
    total_gold = sum(sum(g.values()) for g in golds)

    chunk_texts, chunk_meta = [], []
    for d, text in enumerate(texts):
        for a, b in split_text(text):
            chunk_texts.append(text[a:b])
            chunk_meta.append((d, a))
    print(f"{len(docs)} documents, {len(chunk_texts)} chunks, {total_gold} gold entities "
          f"(dropped, no equivalent label: {dict(dropped)})")

    def conv(chunk_preds, threshold):
        return to_counters(chunk_preds, texts, chunk_meta, threshold, args.strip_titles)

    def row(name, t, counters):
        tp, fp, fn = tally(counters, golds)
        p, r, f = prf(tp, fp, fn)
        print(f"{name:<14}{str(t):>5}{p:>7.3f}{r:>7.3f}{f:>7.3f}   {tp}/{fp}/{fn}")

    model_preds = {}
    for spec in specs:
        name, _, path = spec.partition("=")
        print(f"[{name}] predicting ...", flush=True)
        [model_preds[name]] = ev.run_gliner(path, [chunk_texts], min(thresholds))
    flair_preds = None
    if not args.no_flair:
        ev.FLAIR_MAP.update({"PERCENT": "quantity", "FAC": "location"})
        print("[flair] predicting ...", flush=True)
        flair_preds = ev.run_flair(chunk_texts, args.flair_cpu)

    print("\n=== SUITE RESULTS ===")
    print(f"Title stripping: {'ON' if args.strip_titles else 'off'}. "
          f"One gold entity is worth about {100 / total_gold:.1f} points of recall.\n")
    print(f"{'model':<14}{'thr':>5}{'P':>7}{'R':>7}{'F1':>7}   tp/fp/fn")
    detail = {}
    for name, cp in model_preds.items():
        for t in thresholds:
            counters = conv(cp, t)
            row(name, t, counters)
            if abs(t - args.detail_threshold) < 1e-9:
                detail[name] = counters
    if flair_preds is not None:
        counters = conv(flair_preds, 0.0)
        row("flair", "-", counters)
        detail["flair"] = counters
        if args.ensemble in model_preds:
            ename = f"flair+{args.ensemble}"
            for t in thresholds:
                merged = []
                for fl, gl in zip(flair_preds, model_preds[args.ensemble]):
                    extra = [g for g in gl if g[3] >= t
                             and not any(g[0] < f[1] and f[0] < g[1] for f in fl)]
                    merged.append(fl + extra)
                counters = conv(merged, 0.0)
                row(ename, t, counters)
                if abs(t - args.detail_threshold) < 1e-9:
                    detail[ename] = counters

    print(f"\n=== Boundary-tolerant view at threshold {args.detail_threshold} "
          "(same label, one text contains the other) ===")
    print(f"{'model':<14}{'strictF1':>9}{'lenientP':>10}{'lenientR':>10}{'lenientF1':>11}")
    for name, counters in detail.items():
        strict_f1 = prf(*tally(counters, golds))[2]
        lp, lr, lf = prf(*lenient_tally(counters, golds))
        print(f"{name:<14}{strict_f1:>9.3f}{lp:>10.3f}{lr:>10.3f}{lf:>11.3f}")

    print(f"\n=== Per-label strict F1 at threshold {args.detail_threshold} (support = gold count) ===")
    labels = sorted({k[0] for g in golds for k in g})
    print(f"{'label':<14}{'support':>8}" + "".join(f"{n:>14}" for n in detail))
    for label in labels:
        support = sum(v for g in golds for k, v in g.items() if k[0] == label)
        cells = "".join(f"{prf(*tally(c, golds, label))[2]:>14.2f}" for c in detail.values())
        print(f"{label:<14}{support:>8}{cells}")

    base = detail.get(args.baseline)
    if base is not None:
        print(f"\n=== Gold entities each model gained / lost versus '{args.baseline}' "
              f"(threshold {args.detail_threshold}) ===")
        for name, counters in detail.items():
            if name == args.baseline:
                continue
            gained, lost = [], []
            for doc, g, cm, cb in zip(docs, golds, counters, base):
                gained += [(doc.stem, k, n) for k, n in ((cm & g) - (cb & g)).items()]
                lost += [(doc.stem, k, n) for k, n in ((cb & g) - (cm & g)).items()]
            print(f"\n[{name}] gained {sum(n for *_, n in gained)}, lost {sum(n for *_, n in lost)}")
            for doc, (label, text), n in gained:
                print(f"  + {doc:<14}{label:<13}{text!r} x{n}")
            for doc, (label, text), n in lost:
                print(f"  - {doc:<14}{label:<13}{text!r} x{n}")

    if args.misses in detail:
        print(f"\n=== Missed and spurious for '{args.misses}' (threshold {args.detail_threshold}) ===")
        for doc, g, c in zip(docs, golds, detail[args.misses]):
            for (label, text), n in sorted((g - c).items()):
                print(f"  missed   {doc.stem:<14}{label:<13}{text!r} x{n}")
            for (label, text), n in sorted((c - g).items()):
                print(f"  spurious {doc.stem:<14}{label:<13}{text!r} x{n}")


if __name__ == "__main__":
    main()