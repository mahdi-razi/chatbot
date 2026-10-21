class RagError(Exception):
    pass


class ParseError(RagError):
    """A document could not be parsed."""


class ChunkError(RagError):
    """A parsed document could not be chunked, or had no content."""


class StoreError(RagError):
    """Reading from or writing to the vector store failed."""


class RankError(RagError):
    """The reranker failed."""


class HistoryError(RagError):
    """Chat history could not be read or written."""