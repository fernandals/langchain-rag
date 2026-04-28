from enum import Enum
from typing import Any

class DocumentType(Enum):
    SLIDES = "slides"
    PDF = "pdf"
    UNKNOWN = "unknown"

class RawDocument:
    def __init__(
        self,
        content: str,
        pages: list[dict[str, Any]],
        metadata: dict[str, Any]
    ):
        self.content = content
        self.pages = pages
        self.metadata = metadata

class DocumentChunk:
    def __init__(self, content: str, metadata: dict):
        self.content = content
        self.metadata = metadata

class KnowledgeBase:
    def __init__(self, name: str, retriever):
        self.name = name
        self.retriever = retriever
