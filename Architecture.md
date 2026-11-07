# Architecture and Method

This document describes the system design, the models, the datasets and the evaluation
method of **ner-app**. Installation, usage, results and limitations are in `README.md`.

## 1. Overview

ner-app is a local named entity recognition (NER) system for documents. It ingests TXT,
Markdown, HTML, DOCX and text-based PDF files, normalizes the content into a single text
string, runs one of three interchangeable inference backends, and returns typed results:
entity label, character offsets and a confidence score. After the one-time model
download, all processing is local and no document content leaves the machine.

## 2. System architecture

```text
Document
  -> Ingestion (validation, format-specific extraction)
  -> Normalization (encoding, Unicode form, whitespace)
  -> Chunking (overlapping character windows)
  -> Inference backend: Flair | GLiNER | fine-tuned GLiNER
  -> Post-processing (overlap resolution, global offsets, exact-text grouping)
  -> Typed result (validated schema)
  -> Command-line interface (text / JSON) | local web interface
```

| Layer | Location | Responsibility |
|---|---|---|
| Ingestion | `src/ner_app/ingestion/` | File validation and resource limits; extractors for plain text, Markdown, HTML, DOCX and PDF; encoding detection; Unicode and whitespace normalization; extraction warnings |
| Inference | `src/ner_app/inference/` | Backend implementations (`flair_ner.py`, `gliner_ner.py`), chunking, overlap resolution, backend selection (`factory.py`) |
| Result model | `src/ner_app/models.py` | Validated result schema shared by all interfaces |
| Pipeline | `src/ner_app/pipeline.py` | Orchestrates ingestion and inference into one result |
| Interfaces | `src/ner_app/cli.py`, `web.py`, `web_page.py` | Command-line tool and local web interface over the same pipeline |

Design properties:

- **Separation of concerns.** The interfaces contain no NLP logic; they call the pipeline.
- **Interchangeable backends.** Each backend implements the same prediction interface and
  is selected at start-up (`--backend`). One engine is loaded once per process.
- **Offset convention.** Offsets are zero-based and refer to the extracted, normalized
  text, for example `[0:9]`. They are not positions in the original file.
- **Safeguards.** Default limits are 25 MB per file, 5,000,000 extracted characters and
  1,000 PDF pages, plus archive checks for DOCX. Macros and embedded content are never
  executed. Scanned PDFs are not OCR'd and produce a warning.
- **Chunking.** Long texts are processed in overlapping character windows (Flair: 10,000
  characters with 256 overlap; GLiNER: 1,000 with 200). Overlapping predictions are
  resolved by keeping the longest span, then the higher score.

## 3. Inference backends

### 3.1 Flair (default): `flair/ner-english-ontonotes-large`

- **Architecture.** A sequence tagger built on XLM-RoBERTa-large (24 transformer layers,
  hidden size 1024, 16 attention heads, about 550M parameters, SentencePiece vocabulary
  of 250,000 pieces), fine-tuned with document-level context (the FLERT method).
  Word embeddings use the first sub-token of each word. A linear layer predicts one tag
  per word; there is no CRF and no recurrent layer.
- **Label scheme.** 18 OntoNotes entity types (PERSON, NORP, FAC, ORG, GPE, LOC, PRODUCT,
  EVENT, WORK_OF_ART, LAW, LANGUAGE, DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL,
  CARDINAL), encoded with the BIOES scheme (76 tags including the outside and unknown
  tags and two special markers).
- **Training (model card).** OntoNotes 5.0, 20 epochs, learning rate 5e-6, mini-batch 4,
  AdamW with a one-cycle schedule. The authors report F1 90.93 on OntoNotes.
- **Inference.** Tokenization (segtok), sub-word encoding, contextual embeddings, softmax
  over tags per word, BIOES decoding into spans. The reported score is the model's
  softmax confidence, not a calibrated probability.

### 3.2 GLiNER (optional): `urchade/gliner_medium-v2.1`

- **Architecture.** A bidirectional transformer encoder (209M parameters) that encodes the
  requested entity-type names and the text in one sequence, with a learned `[ENT]` token
  before each type. A feed-forward network builds an embedding for every candidate span
  from its first and last token; another builds an embedding for every entity type. The
  match score is the sigmoid of their dot product. Decoding is greedy over
  non-overlapping spans above a threshold.
- **Consequence.** Entity types are free text supplied at run time (`--labels`), so new
  types need no retraining, but label wording affects results.
- **Training data.** Pile-NER-style data: passages from The Pile annotated by an LLM.
- **Licence.** Apache-2.0 (model card).

### 3.3 Fine-tuned GLiNER (optional): `gliner-nuner`

- **Method.** Continued training of the GLiNER model on NuNER sentences converted to ten
  labels (person, organization, location, date, time, event, product, technology,
  quantity, money).
- **Data preparation.**
  - Tokenization identical to GLiNER's word splitter (verified: 0 mismatches in 2,000 texts).
  - Entity strings aligned as whole-token sequences, labeling every occurrence.
  - NuNER's free-form label names mapped onto the ten classes.
  - A strict filter drops any example containing an unmapped, unaligned or
    conflicting annotation, because unlabeled spans act as negatives during training.
  - Of 1,000,000 source rows, 61,441 remained (train 55,280, validation 3,167,
    test 2,994), split by a SHA-256 hash of the text so no test text is seen in training.
- **Training.** 1,000 optimizer steps on 8,000 examples sampled with a bias toward rarer
  labels; effective batch size 8 (1 x 8 gradient accumulation); learning rate 5e-6
  (encoder) and 1e-5 (other layers); weight decay 0.01; linear decay with 10% warm-up;
  gradient clipping at 1.0; fp32 precision (fp16 was slower and produced non-finite
  gradients on the available GTX 1650, 4 GB).
- **Post-processing.** Decision thresholds tuned on validation data only (global 0.65,
  technology 0.55) and stored in `label_thresholds.json`; honorifics such as Dr. and
  Prof. are removed from PERSON spans.

## 4. Datasets

| Dataset | Content | Role | Licence |
|---|---|---|---|
| NuNER (`numind/NuNER`, `entity` split) | 1,000,000 C4 web sentences annotated by GPT-3.5 with free-form entity types (4.38M annotations and about 200,000 distinct types in the full dataset) | Fine-tuning and held-out evaluation of `gliner-nuner` | MIT (per the NuNER paper) |
| OntoNotes 5.0 (LDC2013T19) | About 1.5M English words of newswire, broadcast, telephone and web text with 18 entity types | Training data of the Flair model (not used directly by this project) | LDC User Agreement |
| Pile-NER | LLM-annotated passages from The Pile | Pre-training data of GLiNER | See the model card |
| Hand-made test documents | 7 documents (76 gold entities), one hard paragraph (29), a news-style text (32) | Real-document evaluation; not included in the repository | Project-owned |

NuNER labels are machine-generated and incomplete, so scores on it measure agreement with
the annotator rather than real-world accuracy.

## 5. Evaluation method

- **Metric.** Strict exact match: a prediction is correct only if span and label both
  match. Micro precision, recall and F1 are reported, with per-label breakdowns.
- **Protocol.** Decision thresholds are selected on the validation split only. The test
  split is used for reporting. Differences below about 0.01 F1 are within sampling noise
  at the sample sizes used.
- **Results** (details and limits in `README.md`):

| Test | Base GLiNER | Fine-tuned GLiNER | Flair |
|---|---|---|---|
| NuNER held-out test, F1 | 0.656 | 0.701 | not comparable (different label scheme) |
| Seven hand-made documents, entities found of 76 | 57 | 61 | 68 |
| Hard paragraph, entities found of 29 | 24 | 24 | 25 |

Flair remains the default engine because it found the most entities on real documents.

## 6. Engineering quality

- 92 automated tests, using fake models, cover ingestion, offsets, chunking, grouping,
  backend wiring, the command-line interface and the web interface.
- A fresh clone installs with `pip install -e ".[web,gliner,dev]"` and passes all tests.
- Model weights and data are excluded from the repository.

## 7. Limitations and future work

- Accuracy depends strongly on text type; weaker on technical text and on Indian names,
  institutions and currency formats.
- The fine-tuned model misses percentages and Indian currency formats that Flair finds,
  while Flair misses some products, events and organizations that GLiNER finds. A
  per-label hybrid is a candidate for future work and needs a fair evaluation on more
  data first.
- Hand-labelled in-domain documents would give a more reliable evaluation set and better
  training data than the machine-labelled NuNER corpus.

## 8. References

- Schweter, S. and Akbik, A. (2020). *FLERT: Document-Level Features for Named Entity
  Recognition.* arXiv:2011.06993.
- Conneau, A. et al. (2019). *Unsupervised Cross-lingual Representation Learning at Scale.*
  arXiv:1911.02116.
- Zaratiana, U. et al. (2023). *GLiNER: Generalist Model for Named Entity Recognition using
  Bidirectional Transformer.* arXiv:2311.08526.
- Bogdanov, S. et al. (2024). *NuNER: Entity Recognition Encoder Pre-training via
  LLM-Annotated Data.* arXiv:2402.15343.
- Weischedel, R. et al. (2013). *OntoNotes Release 5.0.* LDC2013T19.