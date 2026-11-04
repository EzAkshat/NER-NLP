"""Command-line interface: ner-app analyze FILE [--backend flair|gliner] ..."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from ner_app.errors import NerError
from ner_app.inference.factory import BACKENDS, build_ner, parse_labels
from ner_app.inference.flair_ner import DEFAULT_MODEL_ID, InferenceSettings
from ner_app.inference.gliner_ner import GLINER_CHUNK_OVERLAP, GLINER_CHUNK_SIZE
from ner_app.models import AnalysisResult
from ner_app.pipeline import analyze_file

__all__ = ["main", "render_text"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ner-app",
        description="Local named entity recognition for documents.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Extract entities from a document.")
    analyze.add_argument("file", help="Path to a .txt, .md, .html, .docx or .pdf file.")
    analyze.add_argument("--format", choices=["text", "json"], default="text",
                         help="Output format (default: text).")
    analyze.add_argument("--backend", choices=BACKENDS, default="flair",
                         help="NER backend (default: flair).")
    analyze.add_argument("--model", default=None,
                         help="Model ID (default depends on the backend).")
    analyze.add_argument("--labels", default=None,
                         help="gliner only: 'NAME=Prompt,NAME2=Prompt 2'. "
                              "A bare name uses itself as the prompt.")
    analyze.add_argument("--threshold", type=float, default=0.5,
                         help="gliner only: minimum score, 0 to 1 (default: 0.5).")
    analyze.add_argument("--chunk-size", type=int, default=None,
                         help="Characters per inference chunk.")
    analyze.add_argument("--chunk-overlap", type=int, default=None,
                         help="Characters shared between neighbouring chunks.")
    return parser


def render_text(result: AnalysisResult) -> str:
    doc, model = result.document, result.model
    lines = [
        f"Document: {doc.source_name} ({doc.detected_format}, {doc.char_count} characters)",
    ]
    if doc.warnings:
        lines.append("Warnings:")
        lines.extend(f"  [{w.code}] {w.message}" for w in doc.warnings)
    lines.append(f"Model: {model.model_id} ({model.framework} {model.framework_version})")
    lines.append(f"Entities found: {result.total_mentions}")
    for label, count in result.counts_by_label.items():
        lines.append(f"  {label}: {count}")

    if result.mentions:
        lines.append("")
        lines.append("Mentions:")
        for m in result.mentions:
            score = "n/a" if m.score is None else f"{m.score:.3f}"
            lines.append(f"  #{m.id}  {m.label}  {m.text!r}  [{m.start}:{m.end}]  score {score}")

    lines.append("")
    lines.append("Note: scores are model confidence estimates, not probabilities. "
                 "Offsets refer to the extracted, normalized text, not the original file.")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.model is not None and not args.model.strip():
        parser.error("--model cannot be empty")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")

    if args.backend in ("gliner", "gliner-nuner"):
        default_size, default_overlap = GLINER_CHUNK_SIZE, GLINER_CHUNK_OVERLAP
    else:
        default_size, default_overlap = InferenceSettings.chunk_size, InferenceSettings.chunk_overlap
    try:
        settings = InferenceSettings(
            chunk_size=args.chunk_size if args.chunk_size is not None else default_size,
            chunk_overlap=args.chunk_overlap if args.chunk_overlap is not None else default_overlap,
        )
        labels = parse_labels(args.labels) if args.backend == "gliner" else None
    except ValueError as exc:
        parser.error(str(exc))  # exits with status 2

    # Keep library INFO logging out of the way so JSON output stays clean.
    logging.getLogger("flair").setLevel(logging.WARNING)

    try:
        if args.backend in ("gliner", "gliner-nuner"):
            result = analyze_file(
                args.file,
                ner_factory=lambda: build_ner(args.backend, args.model, settings, labels, args.threshold),
            )
        else:
            result = analyze_file(
                args.file, model_id=args.model or DEFAULT_MODEL_ID, settings=settings
            )
    except NerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code

    if args.format == "json":
        print(result.model_dump_json(indent=2))
    else:
        print(render_text(result))
    return 0