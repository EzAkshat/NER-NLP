"""Command-line interface: ner-app analyze FILE [--format text|json] ..."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from ner_app.errors import NerError
from ner_app.inference.flair_ner import DEFAULT_MODEL_ID, InferenceSettings
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
    analyze.add_argument("--model", default=DEFAULT_MODEL_ID,
                         help=f"Flair model ID (default: {DEFAULT_MODEL_ID}).")
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

    if not args.model.strip():
        parser.error("--model cannot be empty")
    try:
        settings = InferenceSettings(
            chunk_size=args.chunk_size if args.chunk_size is not None
            else InferenceSettings.chunk_size,
            chunk_overlap=args.chunk_overlap if args.chunk_overlap is not None
            else InferenceSettings.chunk_overlap,
        )
    except ValueError as exc:
        parser.error(str(exc))  # exits with status 2

    # Keep Flair's INFO logging out of the way so JSON output stays clean.
    logging.getLogger("flair").setLevel(logging.WARNING)

    try:
        result = analyze_file(args.file, model_id=args.model, settings=settings)
    except NerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code

    if args.format == "json":
        print(result.model_dump_json(indent=2))
    else:
        print(render_text(result))
    return 0