"""Fair evaluation of GLiNER checkpoints (and optionally Flair) on the same seeded sample.

Example:
  .venv-train/Scripts/python.exe scripts/evaluate_v2.py ^
      --model base=urchade/gliner_medium-v2.1 --model trial=models/gliner-nuner-trial/checkpoint-20
Add --flair to include Flair (ner-english-ontonotes-large).
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

LABELS = ["person", "organization", "location", "date", "time",
          "event", "product", "technology", "quantity", "money"]
COMMON_LABELS = [l for l in LABELS if l != "technology"]  # labels Flair can also produce
FLAIR_MAP = {"PERSON": "person", "ORG": "organization", "GPE": "location", "LOC": "location",
             "DATE": "date", "TIME": "time", "EVENT": "event", "PRODUCT": "product",
             "MONEY": "money", "QUANTITY": "quantity"}


def load_sample(path, n, seed):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    picked = sorted(random.Random(seed).sample(range(len(rows)), min(n, len(rows))))
    sample = [rows[i] for i in picked]
    texts = [r["text"] for r in sample]
    golds = [{(e["start"], e["end"], e["label"]) for e in r["entities"]} for r in sample]
    return texts, golds


def free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_gliner(path, text_sets, min_threshold):
    import torch
    from gliner import GLiNER

    model = GLiNER.from_pretrained(path)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    outputs = []
    with torch.no_grad():
        for texts in text_sets:
            preds = []
            for i, text in enumerate(texts):
                found = model.predict_entities(text, LABELS, threshold=min_threshold)
                preds.append([(p["start"], p["end"], p["label"], float(p["score"])) for p in found])
                if (i + 1) % 250 == 0:
                    print(f"    {i + 1}/{len(texts)}", flush=True)
            outputs.append(preds)
    del model
    free_gpu()
    return outputs


def run_flair(texts, force_cpu):
    import flair
    import torch
    from flair.data import Sentence
    from flair.models import SequenceTagger

    if force_cpu:
        flair.device = torch.device("cpu")
    tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")
    preds = []
    for i, text in enumerate(texts):
        sentence = Sentence(text)
        tagger.predict(sentence)
        items = []
        for span in sentence.get_spans("ner"):
            label = FLAIR_MAP.get(span.tag)
            if label:
                items.append((span.start_position, span.end_position, label, float(span.score)))
        preds.append(items)
        if (i + 1) % 250 == 0:
            print(f"    {i + 1}/{len(texts)}", flush=True)
    del tagger
    free_gpu()
    return preds


def count(preds, golds, threshold, labels):
    allowed = set(labels)
    tp, fp, fn = Counter(), Counter(), Counter()
    for plist, gold in zip(preds, golds):
        p = {(s, e, l) for s, e, l, sc in plist if sc >= threshold and l in allowed}
        g = {x for x in gold if x[2] in allowed}
        for x in p & g:
            tp[x[2]] += 1
        for x in p - g:
            fp[x[2]] += 1
        for x in g - p:
            fn[x[2]] += 1
    return tp, fp, fn


def prf(t, f, n):
    p = t / (t + f) if t + f else 0.0
    r = t / (t + n) if t + n else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def micro(tp, fp, fn):
    return prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))


def macro_f1(tp, fp, fn, labels):
    scores = [prf(tp[l], fp[l], fn[l])[2] for l in labels if tp[l] + fn[l] > 0]
    return sum(scores) / len(scores) if scores else 0.0


def summarize(preds, golds, threshold, labels):
    tp, fp, fn = count(preds, golds, threshold, labels)
    p, r, f1 = micro(tp, fp, fn)
    return {"threshold": threshold, "precision": p, "recall": r, "f1": f1,
            "tp": sum(tp.values()), "fp": sum(fp.values()), "fn": sum(fn.values()),
            "macro_f1": macro_f1(tp, fp, fn, labels),
            "per_label": {l: {"p": prf(tp[l], fp[l], fn[l])[0], "r": prf(tp[l], fp[l], fn[l])[1],
                              "f1": prf(tp[l], fp[l], fn[l])[2], "support": tp[l] + fn[l]}
                          for l in labels}}


def row(name, s):
    return (f"  {name:<12} thr={s['threshold']:<4} P={s['precision']:.3f} R={s['recall']:.3f} "
            f"F1={s['f1']:.3f} macroF1={s['macro_f1']:.3f}  TP/FP/FN={s['tp']}/{s['fp']}/{s['fn']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[], help="name=path_or_hf_id (repeatable)")
    ap.add_argument("--data", default="data/nuner_v2")
    ap.add_argument("--n", type=int, default=1000, help="sample size for validation and for test")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--grid", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    ap.add_argument("--flair", action="store_true")
    ap.add_argument("--flair-cpu", action="store_true", help="use if Flair runs out of GPU memory")
    args = ap.parse_args()

    grid = [float(x) for x in args.grid.split(",")]
    data = Path(args.data)
    val_texts, val_gold = load_sample(data / "validation.eval.jsonl", args.n, args.seed)
    test_texts, test_gold = load_sample(data / "test.eval.jsonl", args.n, args.seed)
    print(f"Validation sample: {len(val_texts)} texts, {sum(map(len, val_gold))} gold entities")
    print(f"Test sample      : {len(test_texts)} texts, {sum(map(len, test_gold))} gold entities")
    print(f"Seed {args.seed}; same samples for every model.\n")

    results, test_preds = {}, {}
    for spec in args.model:
        name, _, path = spec.partition("=")
        print(f"[{name}] predicting ({path}) ...")
        t0 = time.time()
        val_p, test_p = run_gliner(path, [val_texts, test_texts], min(grid))
        print(f"[{name}] done in {time.time() - t0:.0f}s")

        sweep = {t: summarize(val_p, val_gold, t, LABELS)["f1"] for t in grid}
        best_t = max(sweep, key=sweep.get)  # first maximum
        print("  validation F1 by threshold: " + "  ".join(f"{t}:{f:.3f}" for t, f in sweep.items()))
        print(f"  chosen threshold (validation only): {best_t}")

        results[name] = {"chosen": summarize(test_p, test_gold, best_t, LABELS),
                         "at_0.5": summarize(test_p, test_gold, 0.5, LABELS),
                         "common": summarize(test_p, test_gold, best_t, COMMON_LABELS),
                         "validation_sweep": sweep}
        test_preds[name] = test_p

    if args.flair:
        print("[flair] predicting ...")
        t0 = time.time()
        flair_p = run_flair(test_texts, args.flair_cpu)
        print(f"[flair] done in {time.time() - t0:.0f}s")
        results["flair"] = {"chosen": summarize(flair_p, test_gold, 0.0, LABELS),
                            "common": summarize(flair_p, test_gold, 0.0, COMMON_LABELS)}

    print("\n=== TEST, all 10 labels (GLiNER threshold chosen on validation) ===")
    for name, r in results.items():
        print(row(name, r["chosen"]))
    print("\n=== TEST, GLiNER at fixed threshold 0.5 ===")
    for name, r in results.items():
        if "at_0.5" in r:
            print(row(name, r["at_0.5"]))
    print("\n=== TEST, 9 labels Flair can produce (no 'technology') ===")
    for name, r in results.items():
        print(row(name, r["common"]))

    print("\n=== Per-label at chosen threshold (P / R / F1 / support) ===")
    for name, r in results.items():
        print(f"  [{name}]")
        for label, v in r["chosen"]["per_label"].items():
            print(f"    {label:<13}{v['p']:.3f} / {v['r']:.3f} / {v['f1']:.3f} / {v['support']}")

    Path("results").mkdir(exist_ok=True)
    out = Path("results") / f"eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({"seed": args.seed, "n": args.n, "results": results}, indent=2),
                   encoding="utf-8")
    print(f"\nSaved {out}")


def dump_errors(name, preds, golds, texts, threshold, labels, limit, path):
    rng = random.Random(0)
    lines = [f"Errors for {name} at threshold {threshold} (validation sample)"]
    for label in labels:
        fps, fns = [], []
        for i, (plist, gold) in enumerate(zip(preds, golds)):
            scores = {(s, e, l): sc for s, e, l, sc in plist if sc >= threshold}
            p = {x for x in scores if x[2] == label}
            g = {x for x in gold if x[2] == label}
            fps += [(i, x, scores[x]) for x in p - g]
            fns += [(i, x, None) for x in g - p]
        lines.append(f"\n## {label}: {len(fps)} false positives, {len(fns)} false negatives")
        for tag, items in (("FP", fps), ("FN", fns)):
            for i, (s, e, _), sc in rng.sample(items, min(limit, len(items))):
                ctx = texts[i][max(0, s - 35):e + 35].replace("\n", " ")
                extra = f" score={sc:.2f}" if sc is not None else ""
                lines.append(f'  {tag} "{texts[i][s:e]}"{extra}  ...{ctx}...')
    report = "\n".join(lines)
    print(report)
    Path(path).write_text(report, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[], help="name=path_or_hf_id (repeatable)")
    ap.add_argument("--data", default="data/nuner_v2")
    ap.add_argument("--n", type=int, default=1000, help="sample size for validation and for test")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--grid", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    ap.add_argument("--val-only", action="store_true", help="skip the test set (use when choosing checkpoints)")
    ap.add_argument("--dump-errors", type=int, default=0, help="N sampled errors per label (validation)")
    ap.add_argument("--dump-labels", default="date,time,product,quantity")
    ap.add_argument("--flair", action="store_true")
    ap.add_argument("--flair-cpu", action="store_true", help="use if Flair runs out of GPU memory")
    args = ap.parse_args()

    grid = [float(x) for x in args.grid.split(",")]
    data = Path(args.data)
    val_texts, val_gold = load_sample(data / "validation.eval.jsonl", args.n, args.seed)
    test_texts = test_gold = None
    print(f"Validation sample: {len(val_texts)} texts, {sum(map(len, val_gold))} gold entities")
    if not args.val_only:
        test_texts, test_gold = load_sample(data / "test.eval.jsonl", args.n, args.seed)
        print(f"Test sample      : {len(test_texts)} texts, {sum(map(len, test_gold))} gold entities")
    print(f"Seed {args.seed}; same samples for every model.\n")

    Path("results").mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {}

    for spec in args.model:
        name, _, path = spec.partition("=")
        print(f"[{name}] predicting ({path}) ...")
        t0 = time.time()
        sets = [val_texts] if args.val_only else [val_texts, test_texts]
        outs = run_gliner(path, sets, min(grid))
        print(f"[{name}] done in {time.time() - t0:.0f}s")

        val_p = outs[0]
        sweep = {t: summarize(val_p, val_gold, t, LABELS)["f1"] for t in grid}
        best_t = max(sweep, key=sweep.get)  # first maximum
        print("  validation F1 by threshold: " + "  ".join(f"{t}:{f:.3f}" for t, f in sweep.items()))
        print(f"  chosen threshold (validation only): {best_t}")

        entry = {"chosen_threshold": best_t, "validation_sweep": sweep,
                 "validation": summarize(val_p, val_gold, best_t, LABELS)}
        if not args.val_only:
            test_p = outs[1]
            entry["chosen"] = summarize(test_p, test_gold, best_t, LABELS)
            entry["at_0.5"] = summarize(test_p, test_gold, 0.5, LABELS)
            entry["common"] = summarize(test_p, test_gold, best_t, COMMON_LABELS)
        if args.dump_errors:
            labels = [l for l in args.dump_labels.split(",") if l]
            dump_errors(name, val_p, val_gold, val_texts, best_t, labels, args.dump_errors,
                        Path("results") / f"errors_{name}_{stamp}.txt")
        results[name] = entry

    if args.flair and args.val_only:
        print("\n[note] --flair is ignored with --val-only (Flair is only run on the test sample).")
    elif args.flair:
        print("[flair] predicting ...")
        t0 = time.time()
        flair_p = run_flair(test_texts, args.flair_cpu)
        print(f"[flair] done in {time.time() - t0:.0f}s")
        results["flair"] = {"chosen": summarize(flair_p, test_gold, 0.0, LABELS),
                            "common": summarize(flair_p, test_gold, 0.0, COMMON_LABELS)}

    if args.val_only:
        print("\n=== VALIDATION (threshold chosen on validation itself; for checkpoint selection) ===")
        for name, r in results.items():
            print(row(name, r["validation"]))
        key = "validation"
    else:
        print("\n=== TEST, all 10 labels (GLiNER threshold chosen on validation) ===")
        for name, r in results.items():
            print(row(name, r["chosen"]))
        print("\n=== TEST, GLiNER at fixed threshold 0.5 ===")
        for name, r in results.items():
            if "at_0.5" in r:
                print(row(name, r["at_0.5"]))
        print("\n=== TEST, 9 labels Flair can produce (no 'technology') ===")
        for name, r in results.items():
            print(row(name, r["common"]))
        key = "chosen"

    print("\n=== Per-label at chosen threshold (P / R / F1 / support) ===")
    for name, r in results.items():
        print(f"  [{name}]")
        for label, v in r[key]["per_label"].items():
            print(f"    {label:<13}{v['p']:.3f} / {v['r']:.3f} / {v['f1']:.3f} / {v['support']}")

    out = Path("results") / f"eval_{stamp}.json"
    out.write_text(json.dumps({"seed": args.seed, "n": args.n, "val_only": args.val_only,
                               "results": results}, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()