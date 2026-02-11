from enum import Enum
import os
import re
import fitz
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LCDocument

class DocumentType(Enum):
    SLIDES = "slides"
    TEXT_PDF = "text_pdf"
    #ARTICLE = "article"
    #EXERCISE_LIST = "exercise_list"
    UNKNOWN = "unknown"

# documento inteiro antes de ser dividido em chunks
class IngestedDocument:
    def __init__(
        self,
        text: str,
        metadata: dict,
        pages: list[str] | None = None
    ):
        self.text = text
        self.pages = pages
        self.metadata = metadata

class Chunk:
    def __init__(self, content: str, metadata: dict):
        self.content = content
        self.metadata = metadata

class DocumentLoader:
    def load(self, file_path: str) -> IngestedDocument:
        extension = os.path.splitext(file_path)[1].lower()

        if extension == ".pdf":
            return self._load_pdf(file_path)

        raise ValueError(f"Formato não suportado: {extension}")
    
    def _load_pdf(self, file_path: str) -> IngestedDocument:
        doc = fitz.open(file_path)
        
        num_pages = doc.page_count
        doc_type = self._detect_pdf_type(doc[0])

        full_text = ""
        page_offsets = []

        for page_number, page in enumerate(doc): # type: ignore
            text = page.get_text("text")
            page_offsets.append({
                "page": page_number + 1,
                "start": len(full_text)
            })
            full_text += text + "\n"

        metadata = {
            "source": os.path.basename(file_path),
            "file_path": file_path,
            "num_pages": num_pages,
            "doc_type": doc_type.value
        }

        return IngestedDocument(
            text=full_text,
            pages=page_offsets,
            metadata=metadata
        )

    def _detect_pdf_type(self, page: fitz.Page) -> DocumentType:
        width = page.mediabox_size[0]
        height = page.mediabox_size[1]

        return (
            DocumentType.SLIDES
            if width / height > 1.1
            else DocumentType.TEXT_PDF
        )

class SectionDetector:
    SECTION_PATTERNS = [
        # 1. Introduction / 2.3 Client-Server Architecture
        r"^(\d+(\.\d+)*)\s+.+",

        # CHAPTER 4 - Architectural Styles
        r"^CHAPTER\s+\d+.*",

        # TITLES
        r"^[A-Z][A-Z\s]{5,}$"
    ]

    def detect(self, text: str) -> list[dict]:
        sections = []
        lines = text.splitlines()

        char_index = 0  # posição absoluta no texto

        for line in lines:
            stripped = line.strip()

            for pattern in self.SECTION_PATTERNS:
                if re.match(pattern, stripped):
                    sections.append({
                        "title": stripped,
                        "start_index": char_index
                    })
                    break

            # +1 por causa do '\n'
            char_index += len(line) + 1

        return sections

# Parsers por tipo de documento
class BaseParser:
    def parse(self, doc: IngestedDocument) -> IngestedDocument:
        raise NotImplementedError
    
class PDFTextParser(BaseParser):
    def __init__(self):
        self.section_detector = SectionDetector()

    def parse(self, doc: IngestedDocument) -> IngestedDocument:
        sections = self.section_detector.detect(doc.text)
        doc.metadata["sections"] = sections
        return doc

class SectionAwarePDFParser(BaseParser):
    SECTION_REGEX = re.compile(
        r"(?m)^(\d+(?:\.\d+)*)\s{2,}([A-Z][^\n]{3,80})$"
    )

    def parse(self, doc: IngestedDocument) -> IngestedDocument:
        text = doc.text
        sections = []

        matches = list(self.SECTION_REGEX.finditer(text))

        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            sections.append({
                "section_id": m.group(1),
                "title": m.group(2).strip(),
                "start": start,
                "end": end
            })

        doc.metadata["sections"] = sections
        return doc

class SlidePDFParser(BaseParser):
    def parse(self, doc: IngestedDocument) -> IngestedDocument:
        slides = []

        for i, page_text in enumerate(doc.pages): # type: ignore
            title, body = self._extract_slide_structure(page_text)
            slides.append({
                "page": i + 1,
                "title": title,
                "body": body
            })

        doc.metadata["slides"] = slides
        return doc

    def _extract_slide_structure(self, text: str):
        lines = [l.strip() for l in text.splitlines() if l.strip()]

        if not lines:
            return None, ""

        title = lines[0]
        body = "\n".join(lines[1:])

        return title, body

class ParserFactory:
    @staticmethod
    def get_parser(doc_type: DocumentType) -> BaseParser:
        if doc_type == DocumentType.SLIDES.value:
            return SlidePDFParser()

        if doc_type == DocumentType.TEXT_PDF.value:
            #return PDFTextParser()
            return SectionAwarePDFParser()

        raise ValueError(f"Parser não suportado para {doc_type}")

# Splitters por tipo de documento
class BaseSplitter:
    def split(self, doc: IngestedDocument) -> list[Chunk]:
        raise NotImplementedError

# usado para apostilas, livros, pdfs longos
# chunk por token, overlap, respeita capitulos
class RecursiveTextSplitter(BaseSplitter):
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )

    def split(self, doc: IngestedDocument) -> list[Chunk]:
        lc_doc = LCDocument(
            page_content=doc.text,
            metadata={}
        )

        lc_chunks = self.splitter.split_documents([lc_doc])

        chunks: list[Chunk] = []

        for i, c in enumerate(lc_chunks):
            metadata = {
                "chunk_id": i,
                "doc_type": doc.metadata["doc_type"].value
                if hasattr(doc.metadata["doc_type"], "value")
                else doc.metadata["doc_type"],
                "start_index": c.metadata.get("start_index"),
                "chunk_role": "text"
            }

            chunks.append(
                Chunk(
                    content=c.page_content,
                    metadata=metadata
                )
            )

        return chunks

# usado para slides (1 slide = 1 chunk)
# 0 overlap
class SlideSplitter(BaseSplitter):
    def split(self, doc: IngestedDocument) -> list[Chunk]:
        slides = doc.metadata.get("slides", [])
        chunks: list[Chunk] = []

        for i, slide in enumerate(slides):
            content_parts = []

            if slide.get("title"):
                content_parts.append(slide["title"])

            if slide.get("body"):
                content_parts.append(slide["body"])

            content = "\n".join(content_parts).strip()

            metadata = {
                "chunk_id": i,
                "doc_type": doc.metadata["doc_type"].value
                if hasattr(doc.metadata["doc_type"], "value")
                else doc.metadata["doc_type"],
                "page_start": slide["page"],
                "page_end": slide["page"],
                "chunk_role": "slide"
            }

            chunks.append(Chunk(content=content, metadata=metadata))

        return chunks

class SplitterFactory:
    @staticmethod
    def get_splitter(doc_type: DocumentType) -> BaseSplitter:
        if doc_type == DocumentType.SLIDES.value:
            return SlideSplitter()

        if doc_type == DocumentType.TEXT_PDF.value:
            return RecursiveTextSplitter()

        raise ValueError(f"Splitter não suportado para {doc_type}")

class MetadataEnricher:
    def enrich(
        self,
        chunk: Chunk,
        doc: IngestedDocument
    ) -> Chunk:

        enriched_metadata = {}

        # -------- Documento --------
        enriched_metadata["source"] = doc.metadata.get("source")
        enriched_metadata["file_path"] = doc.metadata.get("file_path")
        enriched_metadata["doc_type"] = (
            doc.metadata["doc_type"].value
            if hasattr(doc.metadata["doc_type"], "value")
            else doc.metadata["doc_type"]
        )
        enriched_metadata["num_pages"] = doc.metadata.get("num_pages")

        # -------- Chunk --------
        enriched_metadata.update(chunk.metadata)

        # -------- PDF texto --------
        if enriched_metadata["doc_type"] == DocumentType.TEXT_PDF.value:
            sections = doc.metadata.get("sections", [])
            start_index = enriched_metadata.get("start_index")

            if start_index is not None and sections:
                section = self._find_section_for_index(start_index, sections)
                if section:
                    enriched_metadata["section_title"] = section["title"]

        # -------- Slides --------
        if enriched_metadata["doc_type"] == DocumentType.SLIDES.value:
            slides = doc.metadata.get("slides", [])
            chunk_id = chunk.metadata.get("chunk_id")

            if chunk_id is not None and chunk_id < len(slides):
                slide = slides[chunk_id]
                enriched_metadata["slide_title"] = slide.get("title")

        return Chunk(
            content=chunk.content,
            metadata=enriched_metadata
        )

    def _find_section_for_index(
        self,
        start_index: int,
        sections: list[dict]
    ) -> dict | None:
        current = None

        for section in sections:
            if section["start_index"] <= start_index:
                current = section
            else:
                break

        return current

# futura implementação de armazenamento em banco de dados
class StorageAdapter:
    def store(self, chunks: list[Chunk]):
        raise NotImplementedError


class IngestionPipeline:
    def __init__(
        self,
        loader: DocumentLoader,
        enricher: MetadataEnricher
        # storage: StorageAdapter | None = None
    ) -> None:
        self.loader = loader
        self.enricher = enricher
        # self.storage = storage

    def ingest(self, file_path: str) -> list[Chunk]:
        print("\n==============================")
        print("[INGESTION] Starting ingestion")
        print(f"[INGESTION] File: {file_path}")

        print("[INGESTION] Loading document...")
        doc = self.loader.load(file_path)

        print("==== PRIMEIROS 1000 CHARS ====")
        print(doc.text[:1000])

        print("\n==== QUEBRAS DE LINHA (repr) ====")
        print(repr(doc.text[:500]))

        doc_type = doc.metadata.get("doc_type")
        print(f"[INGESTION] Detected document type: {doc_type}")

        parser = ParserFactory.get_parser(doc_type) # type: ignore
        print(f"[INGESTION] Using parser: {parser.__class__.__name__}")
        doc = parser.parse(doc)

        print(doc.metadata)

        splitter = SplitterFactory.get_splitter(doc_type) # type: ignore
        print(f"[INGESTION] Using splitter: {splitter.__class__.__name__}")
        chunks = splitter.split(doc)

        print(f"[INGESTION] Generated {len(chunks)} chunks")

        print("[INGESTION] Enriching metadata...")
        enriched_chunks: list[Chunk] = []

        for c in chunks:
            enriched = self.enricher.enrich(c, doc)
            enriched_chunks.append(enriched)

        print("[INGESTION] Metadata enrichment completed")

        # if self.storage:
        #     print("[INGESTION] Storing chunks...")
        #     self.storage.store(enriched_chunks)

        print("[INGESTION] Ingestion finished successfully")
        print("==============================\n")

        return enriched_chunks

def main():
    loader = DocumentLoader()
    enricher = MetadataEnricher()

    pipeline = IngestionPipeline(
        loader=loader,
        enricher=enricher
    )

    # PDF texto
    chunks_text = pipeline.ingest("pdfs/SAIA-Chapter12.pdf")

    print(f"Metadata do primeiro chunk:\n{chunks_text[0].metadata}")
    print(f"Content do primeiro chunk:\n{chunks_text[0].content[:200]}")
    print("-----")
    print(f"Metadata do segundo chunk:\n{chunks_text[1].metadata}")
    print(f"Content do segundo chunk:\n{chunks_text[1].content[:200]}")


    # Slides
    #chunks_slides = pipeline.ingest("pdfs/SAIA-Chapter2-slide.pdf")

    #(chunks_slides[0].metadata)
    #print(chunks_slides[1].metadata)
    #print(chunks_slides[1].content[:200])  

if __name__ == "__main__":
    main()