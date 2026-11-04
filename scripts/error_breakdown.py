"""Classify every disagreement between a model and the validation gold labels.

For each prediction that is not an exact (span + label) match:
  wrong_label          same span, but gold has a different label  -> label-convention conflict
  boundary             overlaps a gold span with the SAME label    -> boundary disagreement
  overlap_other_label  overlaps a gold span with a different label
  no_gold_overlap      touches no gold span at all -> a model error OR a missing gold annotation
Gold entities that were not matched exactly are classified the same way
(not_found = no prediction touches them).
"""

import argparse
from collections import Counter
from pathlib import Path

from evaluate_v2 import LABELS, load_sample, run_gliner


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="urchade/gliner_medium-v2.1")
    ap.add_argument("--data", default="data/nuner_v2")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--confident", type=float, default=0.85)
    args = ap.parse_args()

    texts, golds = load_sample(Path(args.data) / "validation.eval.jsonl", args.n, args.seed)
    print(f"Predicting {len(texts)} validation texts with {args.model} (threshold {args.threshold}) ...")
    [preds] = run_gliner(args.model, [texts], args.threshold)

    fp_cats = {l: Counter() for l in LABELS}
    fn_cats = {l: Counter() for l in LABELS}
    confusion = Counter()
    confident_total, confident_gap = Counter(), Counter()

    for plist, gold in zip(preds, golds):
        exact = {(s, e, l) for s, e, l, _ in plist}
        for s, e, l, score in plist:
            if (s, e, l) in gold:
                continue
            near = [g for g in gold if overlaps((s, e), g)]
            same_span = [g for g in near if g[0] == s and g[1] == e]
            if same_span:
                cat = "wrong_label"
                confusion[(same_span[0][2], l)] += 1
            elif any(g[2] == l for g in near):
                cat = "boundary"
            elif near:
                cat = "overlap_other_label"
            else:
                cat = "no_gold_overlap"
            fp_cats[l][cat] += 1
            if score >= args.confident:
                confident_total[l] += 1
                confident_gap[l] += cat == "no_gold_overlap"
        for g in gold - exact:
            near = [p for p in plist if overlaps((p[0], p[1]), g)]
            if any(p[0] == g[0] and p[1] == g[1] for p in near):
                cat = "wrong_label"
            elif any(p[2] == g[2] for p in near):
                cat = "boundary"
            elif near:
                cat = "overlap_other_label"
            else:
                cat = "not_found"
            fn_cats[g[2]][cat] += 1

    def table(title, cats, cols):
        print(f"\n{title}")
        print(f"  {'label':<13}{'total':>6}" + "".join(f"{c:>21}" for c in cols))
        grand = Counter()
        for label in LABELS:
            c = cats[label]
            grand.update(c)
            print(f"  {label:<13}{sum(c.values()):>6}" + "".join(f"{c[k]:>21}" for k in cols))
        print(f"  {'ALL':<13}{sum(grand.values()):>6}" + "".join(f"{grand[k]:>21}" for k in cols))

    table("FALSE POSITIVES by predicted label",
          fp_cats, ["wrong_label", "boundary", "overlap_other_label", "no_gold_overlap"])
    table("FALSE NEGATIVES by gold label",
          fn_cats, ["wrong_label", "boundary", "overlap_other_label", "not_found"])

    print("\nMost common label conflicts on the SAME span (gold -> model):")
    for (gold_label, pred_label), n in confusion.most_common(12):
        print(f"  {gold_label:<13}-> {pred_label:<13}{n:>5}")

    print(f"\nConfident predictions (score >= {args.confident}) touching NO gold span (likely missing gold):")
    for label in LABELS:
        print(f"  {label:<13}{confident_gap[label]:>5} of {confident_total[label]:>5} confident false positives")


if __name__ == "__main__":
    main()