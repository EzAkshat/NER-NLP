# ner-app

A local, privacy-focused command-line tool for English named entity recognition (NER).
It reads a document, extracts its text, runs a Flair NER model, and reports each entity
with its label and exact character offsets. Nothing is sent to a remote service once the
model has been downloaded.

## Requirements

- Python 3.13 (developed and tested on Python 3.13.7, Windows, CPU only).
  `pyproject.toml` allows Python 3.10 and later, but other versions are untested.
- About 2 GB of disk space for the model, plus the PyTorch install.
- Flair 0.15.x (tested with 0.15.1) and PyTorch (tested with 2.14.1).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows (Git Bash): source .venv/Scripts/activate
pip install -e .
pip install -e ".[dev]"          # optional: adds pytest
```

### First run and model download

The default model is `flair/ner-english-ontonotes-large`. It is downloaded
automatically the first time you analyze a document and cached under `~/.flair`
(about 2.1 GB on the machine it was tested on). Later runs reuse the cache.

On a CPU-only machine, expect roughly 30 seconds for model loading plus analysis of a
small file. The model is loaded once per run, so the cost is mostly fixed.

## Usage

```bash
ner-app analyze FILE [--format text|json] [--model MODEL_ID]
                     [--chunk-size N] [--chunk-overlap N]
```

`python -m ner_app analyze FILE` works too.

Example:

```text
$ ner-app analyze samples/sample.html
Document: sample.html (html, 137 characters)
Model: flair/ner-english-ontonotes-large (Flair 0.15.1)
Entities found: 7
  DATE: 2
  GPE: 1
  MONEY: 1
  ORG: 1
  PERCENT: 1
  PERSON: 1

Mentions:
  #0  DATE  'Quarterly'  [0:9]  score 0.998
  #1  PERSON  'Jane Smith'  [18:28]  score 1.000
  ...
```

Note that `'Quarterly'` is a model mistake (a heading read as a date). Review the output
before relying on it.

### JSON output

`--format json` writes the full structured result to stdout. Logs and errors go to
stderr, so the output can be piped:

```bash
ner-app analyze report.docx --format json > result.json
```

The JSON contains the document info (including any extraction warnings), model info,
every mention, and exact-text groupings of repeated entities. Groups never replace the
original mentions.

### Advanced options

Long documents are processed in overlapping character chunks.
`--chunk-size` (default 10,000) and `--chunk-overlap` (default 256) control this.
The overlap must be smaller than the chunk size. `--model` selects another Flair NER
model ID.

## Supported formats

`.txt`, `.md`, `.html`, `.docx`, and `.pdf`.

Default safeguards (the code reads them from `Limits`): files up to 25 MB, 5,000,000
extracted characters, and 1,000 PDF pages, plus archive checks for DOCX files.
These are starting values, not measured limits.

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error (for example, model loading or inference failed) |
| 2 | Problem with the input document, or invalid command-line arguments |
| 3 | A resource limit was exceeded |

## Limitations

- **Offsets refer to the extracted, normalized text**, not to the original file.
  They are not PDF coordinates or DOCX byte positions.
- **Scores are not probabilities.** They are Flair's confidence estimates and can be
  high for wrong predictions.
- **Model accuracy is not guaranteed.** The automated tests use a fake tagger. The real
  model has only been checked on small samples.
- **Chunking is by fixed character windows**, not word or sentence boundaries. Where
  chunks overlap, overlapping predictions are resolved by keeping the longest span,
  then the higher score. An entity longer than `--chunk-overlap` can still be split.
- **Scanned PDFs are not OCR'd.** They produce a warning and no text.
- English only, using the OntoNotes label set.

## Development

```bash
python -m pytest -q
```