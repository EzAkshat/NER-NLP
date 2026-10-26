"""Expected, user-facing errors. The CLI will map these to exit codes and friendly messages."""


class NerError(Exception):
    """Base class for all expected application errors."""

    exit_code = 1


class DocumentError(NerError):
    """Something is wrong with the input document."""

    exit_code = 2


class DocumentNotFoundError(DocumentError):
    pass


class UnsupportedFormatError(DocumentError):
    pass


class UnreadableDocumentError(DocumentError):
    pass


class EmptyDocumentError(DocumentError):
    pass


class ExtractionError(DocumentError):
    pass


class ResourceLimitError(DocumentError):
    exit_code = 3