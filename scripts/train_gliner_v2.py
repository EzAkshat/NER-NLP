"""Controlled GLiNER fine-tuning on the prepared NuNER v2 data.

Smoke test first, e.g.:
  .venv-train/Scripts/python.exe scripts/train_gliner_v2.py --run smoke16 --max-steps 30 --save-steps 15
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path

import torch
from gliner import GLiNER

BASE_MODEL = "urchade/gliner_medium-v2.1"
REQUIRED_KWARGS = {"train_dataset", "eval_dataset", "output_dir"}


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pick_subset(examples, n, seed, mode):
    """Choose n examples. 'balanced' favours examples containing rarer labels."""
    if n >= len(examples):
        return list(examples)
    rng = random.Random(seed)
    if mode == "random":
        return rng.sample(examples, n)
    freq = Counter()
    for ex in examples:
        freq.update({label for _, _, label in ex["ner"]})
    # Weighted sampling without replacement (Efraimidis-Spirakis, in log space).
    keys = []
    for ex in examples:
        weight = max(1.0 / math.sqrt(freq[label]) for _, _, label in ex["ner"])
        keys.append(math.log(rng.random() or 1e-300) / weight)
    order = sorted(range(len(examples)), key=lambda i: keys[i], reverse=True)
    return [examples[i] for i in order[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name; output goes to models/gliner-v2/<run>")
    ap.add_argument("--data", default="data/nuner_v2")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--save-steps", type=int, default=250)
    ap.add_argument("--keep", type=int, default=4, help="checkpoints to keep")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--others-lr", type=float, default=1e-5)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp32")
    ap.add_argument("--subset", type=int, default=None,
                    help="training examples to use (default: max-steps * accum = one pass, no repeats)")
    ap.add_argument("--sampling", choices=["balanced", "random"], default="balanced")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-file", default="train.jsonl")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Training requires your NVIDIA GPU.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = Path(args.data)
    train_all = load_jsonl(data / args.train_file)
    val_all = load_jsonl(data / "validation.jsonl")
    n_train = args.subset or args.max_steps * args.accum
    train = pick_subset(train_all, n_train, args.seed, args.sampling)
    val_small = random.Random(args.seed).sample(val_all, min(200, len(val_all)))

    subset_labels = Counter(l for ex in train for _, _, l in ex["ner"])
    print(f"Training examples: {len(train):,} of {len(train_all):,} ({args.sampling} sampling)")
    print("Entity counts in subset: " + ", ".join(f"{k}={v:,}" for k, v in subset_labels.most_common()))

    out_dir = Path("models/gliner-v2") / args.run
    out_dir.mkdir(parents=True, exist_ok=True)

    model = GLiNER.from_pretrained(BASE_MODEL)

    kwargs = dict(
        train_dataset=train,
        eval_dataset=val_small,  # kept for the API; no evaluation runs during training
        output_dir=str(out_dir),
        max_steps=args.max_steps,
        learning_rate=args.lr,
        others_lr=args.others_lr,
        weight_decay=0.01,
        others_weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.accum,
        fp16=(args.precision == "fp16"),
        bf16=False,
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=args.keep,
        save_only_model=True,
        seed=args.seed,
        report_to="none",
    )

    removed = []
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    trainer = None
    for _ in range(6):
        try:
            trainer = model.train_model(**kwargs)
            break
        except TypeError as exc:
            match = re.search(r"unexpected keyword argument '(\w+)'", str(exc))
            name = match.group(1) if match else None
            if name is None or name not in kwargs or name in REQUIRED_KWARGS:
                raise
            print(f"\n[note] this GLiNER version does not accept '{name}'; retrying without it.")
            removed.append(name)
            kwargs.pop(name)
    else:
        raise RuntimeError("Too many unsupported arguments; send me the output.")
    elapsed = time.time() - start

    (out_dir / "train_config.json").write_text(
        json.dumps({"args": vars(args), "removed_kwargs": removed,
                    "subset_entity_counts": dict(subset_labels)}, indent=2),
        encoding="utf-8",
    )

    print("\n=== Training summary ===")
    print(f"Precision: {args.precision} | steps: {args.max_steps} | wall time: {elapsed:.0f}s "
          f"({elapsed / args.max_steps:.1f}s/step)")
    print(f"Peak VRAM allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    if removed:
        print(f"Arguments not supported by this version (dropped): {removed}")

    history = getattr(getattr(trainer, "state", None), "log_history", None)
    if history:
        norms = [h["grad_norm"] for h in history if "grad_norm" in h]
        bad = [i for i, n in enumerate(norms) if not math.isfinite(float(n))]
        losses = [float(h["loss"]) for h in history if "loss" in h and math.isfinite(float(h["loss"]))]
        print(f"Logged steps: {len(norms)} | non-finite grad norms: {len(bad)} at steps {bad[:30]}")
        if len(losses) >= 20:
            print(f"Mean loss, first 10 logged: {sum(losses[:10]) / 10:.3f} | last 10: {sum(losses[-10:]) / 10:.3f}")
    else:
        print("Could not read the trainer's log history (train_model did not return a trainer).")

    print("Checkpoints:", sorted(p.name for p in out_dir.glob("checkpoint-*")))


if __name__ == "__main__":
    main()