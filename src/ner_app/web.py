"""Local web interface: upload a document, see highlighted entities.

Binds to 127.0.0.1 only. Nothing is stored; uploads live in a temp dir for one request.
"""

from __future__ import annotations

import argparse
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from ner_app.errors import NerError
from ner_app.inference.flair_ner import FlairNER
from ner_app.ingestion import DEFAULT_LIMITS, Limits, extract_document
from ner_app.pipeline import analyze_document
from ner_app.web_page import PAGE

__all__ = ["create_app", "main"]

# NerError.exit_code -> HTTP status
_STATUS = {2: 422, 3: 413}


def create_app(ner: FlairNER | None = None, limits: Limits = DEFAULT_LIMITS) -> FastAPI:
    """Build the app. If `ner` is None, the real model is loaded once at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.ner is None:
            print("Loading NER model (this can take 30-40 seconds)...", flush=True)
            app.state.ner = FlairNER()
            print("Model ready.", flush=True)
        yield

    app = FastAPI(title="ner-app", lifespan=lifespan)
    app.state.ner = ner
    lock = Lock()  # one inference at a time

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

            try:
                document = extract_document(path, limits)
                with lock:
                    result = analyze_document(document, app.state.ner)
            except NerError as exc:
                raise HTTPException(_STATUS.get(exc.exit_code, 500), str(exc)) from exc

        # Show the real file name, not the temp name.
        doc_info = result.document.model_copy(update={"source_name": name})
        result = result.model_copy(update={"document": doc_info})
        return JSONResponse({"text": document.text, "result": result.model_dump(mode="json")})

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(prog="ner-app-web", description="Local web interface.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print(f"Open http://127.0.0.1:{args.port} in your browser once the model is ready.", flush=True)
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()