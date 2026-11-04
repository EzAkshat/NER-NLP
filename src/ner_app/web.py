"""Local web interface: upload a document or paste text, see highlighted entities.

Binds to 127.0.0.1 only. Nothing is stored; input lives in a temp dir for one request.
"""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from ner_app.errors import NerError
from ner_app.inference.factory import BACKENDS, build_ner, parse_labels
from ner_app.inference.flair_ner import FlairNER
from ner_app.ingestion import DEFAULT_LIMITS, Limits, extract_document
from ner_app.pipeline import analyze_document
from ner_app.web_page import PAGE

__all__ = ["create_app", "main"]

# NerError.exit_code -> HTTP status
_STATUS = {2: 422, 3: 413}


class TextInput(BaseModel):
    text: str


def create_app(
    ner: Any | None = None,
    limits: Limits = DEFAULT_LIMITS,
    ner_factory: Callable[[], Any] | None = None,
) -> FastAPI:
    """Build the app. If `ner` is None, the model is built once at startup.

    `ner_factory` builds the model at startup (default: the Flair model).
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.ner is None:
            print("Loading NER model (this can take 30-40 seconds)...", flush=True)
            app.state.ner = (ner_factory or FlairNER)()
            print("Model ready.", flush=True)
        yield

    app = FastAPI(title="ner-app", lifespan=lifespan)
    app.state.ner = ner
    lock = Lock()  # one inference at a time

    def run(path: Path, display_name: str) -> JSONResponse:
        """Extract and analyze one file on disk, then build the response."""
        try:
            document = extract_document(path, limits)
            with lock:
                result = analyze_document(document, app.state.ner)
        except NerError as exc:
            raise HTTPException(_STATUS.get(exc.exit_code, 500), str(exc)) from exc

        # Show the real name, not the temp file name.
        doc_info = result.document.model_copy(update={"source_name": display_name})
        result = result.model_copy(update={"document": doc_info})
        return JSONResponse({"text": document.text, "result": result.model_dump(mode="json")})

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE

    @app.post("/analyze")
    def analyze(file: UploadFile = File(...)):
        name = Path(file.filename or "upload").name or "upload"
        suffix = Path(name).suffix.lower()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"upload{suffix}"
            size = 0
            with path.open("wb") as out:
                while chunk := file.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > limits.max_file_bytes:
                        raise HTTPException(413, "File is larger than the allowed limit.")
                    out.write(chunk)
            return run(path, name)

    @app.post("/analyze-text")
    def analyze_text(payload: TextInput):
        data = payload.text.encode("utf-8", errors="replace")
        if len(data) > limits.max_file_bytes:
            raise HTTPException(413, "Text is larger than the allowed limit.")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pasted.txt"
            path.write_bytes(data)  # bytes, so Windows does not rewrite newlines
            return run(path, "pasted text")

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(prog="ner-app-web", description="Local web interface.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend", choices=BACKENDS, default="flair")
    parser.add_argument("--model", default=None)
    parser.add_argument("--labels", default=None, help="gliner only: 'NAME=Prompt,...'")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    try:
        labels = parse_labels(args.labels) if args.backend == "gliner" else None
    except ValueError as exc:
        parser.error(str(exc))

    def factory():
        return build_ner(args.backend, args.model, None, labels, args.threshold)

    print(f"Open http://127.0.0.1:{args.port} in your browser once the model is ready.", flush=True)
    uvicorn.run(create_app(ner_factory=factory), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()