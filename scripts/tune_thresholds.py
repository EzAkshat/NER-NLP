"""Tune GLiNER decision thresholds on the VALIDATION split and save them next to the model.

Writes <model>/label_thresholds.json, read automatically by the 'gliner-nuner' backend.
The TEST split is scored once at the end as a sanity check only; nothing is tuned on it.

Run: .venv-train/Scripts/python.exe scripts/tune_thresholds.py --model models/gliner-nuner-v1
"""

import argparse
import json
from datetime import date
from pathlib import Path

import evaluate_v2 as ev

GRID = [round(0.2 + 0.05 * i, 2) for i in range(15)]  # 0.20 .. 0.90
MIN_SUPPORT = 100  # a label needs this many validation entities to get its own threshold
MIN_GAIN = 0.01    # ...and must beat the global threshold's F1 by this much


def label_f1(preds, golds, threshold, label):
    tp, fp, fn = ev.count(preds, golds, threshold, [label])
    return ev.prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))[2]


def combined(preds, golds, default, per_label):
    tp = fp = fn = 0
    for label in ev.LABELS:
        a, b, c = ev.count(preds, golds, per_label.get(label, default), [label])
        tp, fp, fn = tp + sum(a.values()), fp + sum(b.values()), fn + sum(c.values())
    return ev.prf(tp, fp, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model folder, e.g. models/gliner-nuner-v1")
    ap.add_argument("--data", default="data/nuner_v2")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    model_dir = Path(args.model)
    if not model_dir.is_dir():
        raise SystemExit(f"{model_dir} is not a folder. Copy the checkpoint there first.")

    data = Path(args.data)
    val_texts, val_gold = ev.load_sample(data / "validation.eval.jsonl", 10**9, args.seed)
    test_texts, test_gold = ev.load_sample(data / "test.eval.jsonl", 10**9, args.seed)
    print(f"Validation: {len(val_texts)} texts | test: {len(test_texts)} texts. "
          f"Predicting both (a few minutes) ...", flush=True)
    val_p, test_p = ev.run_gliner(str(model_dir), [val_texts, test_texts], min(GRID))

    global_t = max(GRID, key=lambda t: combined(val_p, val_gold, t, {})[2])
    print(f"\nGlobal threshold chosen on validation: {global_t}")

    per_label = {}
    print(f"\n{'label':<13}{'support':>8}{'F1@global':>11}{'best_thr':>10}{'F1@best':>9}  adopted")
    for label in ev.LABELS:
        support = sum(1 for g in val_gold for x in g if x[2] == label)
        base_f1 = label_f1(val_p, val_gold, global_t, label)
        best_t = max(GRID, key=lambda t: label_f1(val_p, val_gold, t, label))
        best_f1 = label_f1(val_p, val_gold, best_t, label)
        adopt = support >= MIN_SUPPORT and best_t != global_t and best_f1 - base_f1 >= MIN_GAIN
        if adopt:
            per_label[label] = best_t
        print(f"{label:<13}{support:>8}{base_f1:>11.3f}{best_t:>10}{best_f1:>9.3f}  {'YES' if adopt else '-'}")

    def show(name, preds, golds):
        g = combined(preds, golds, global_t, {})
        pl = combined(preds, golds, global_t, per_label)
        print(f"  {name:<11} global only: P={g[0]:.3f} R={g[1]:.3f} F1={g[2]:.3f} | "
              f"per-label: P={pl[0]:.3f} R={pl[1]:.3f} F1={pl[2]:.3f}")
        return g[2], pl[2]

    print("\nMicro scores:")
    show("validation", val_p, val_gold)
    test_global, test_per_label = show("test", test_p, test_gold)

    if per_label and test_per_label < test_global:
        print("\nPer-label thresholds did worse than a single threshold on test; saving the single one only.")
        per_label = {}

    out = model_dir / "label_thresholds.json"
    out.write_text(json.dumps({
        "default": global_t,
        "labels": per_label,
        "tuned_on": f"validation split, {len(val_texts)} texts, seed {args.seed}",
        "date": date.today().isoformat(),
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out}\n{out.read_text(encoding='utf-8')}")


if __name__ == "__main__":
    main()