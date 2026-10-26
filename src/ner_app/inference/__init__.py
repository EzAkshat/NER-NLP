"""Named entity recognition inference."""

from ner_app.inference.flair_ner import FlairNER, InferenceError, InferenceSettings

__all__ = ["FlairNER", "InferenceError", "InferenceSettings"]