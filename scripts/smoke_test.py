"""Phase 0: inspect what the model actually returns. Nothing here is production code."""
import sys
import time
from importlib.metadata import version

import torch
import flair
from flair.data import Sentence
from flair.models import SequenceTagger

MODEL_ID = "flair/ner-english-ontonotes-large"

TEXT = (
    "Jane Smith joined Example Technologies in Berlin on 3 March 2021 for $2.5 million. "
    "She told the BBC that Microsoft Corporation and Microsoft would open a new office.\n\n"
    "Café Müller hosted the German delegation on Tuesday; about 40% attended."
)

print("Python :", sys.version.split()[0])
print("flair  :", version("flair"))
print("torch  :", torch.__version__)
print("device :", flair.device)

t0 = time.time()
tagger = SequenceTagger.load(MODEL_ID)
print(f"model loaded in {time.time() - t0:.1f}s")
print("label dictionary:", sorted(tagger.label_dictionary.get_items()))

sentence = Sentence(TEXT)
t0 = time.time()
tagger.predict(sentence)
print(f"inference took {time.time() - t0:.2f}s\n")

print(f"{'text':28} {'label':12} {'start':>5} {'end':>5} {'score':>7}  offset_ok")
for span in sentence.get_spans("ner"):
    start, end = span.start_position, span.end_position
    ok = TEXT[start:end] == span.text
    print(f"{span.text[:27]:28} {span.tag:12} {start:5} {end:5} {span.score:7.4f}  {ok}")
    if not ok:
        print(f"   !! TEXT[start:end] = {TEXT[start:end]!r}")