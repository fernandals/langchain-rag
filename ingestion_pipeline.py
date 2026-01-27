from enum import Enum
import os
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
        self.pages = pages # util pra slides
        self.metadata = metadata

class Chunk:
    def __init__(self, content: str, metadata: dict):
        self.content = content
        self.metadata = metadata

class DocumentLoader:
    def load(self, file_path: str) -> IngestedDocument:
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return self._load_pdf(file_path)

        raise ValueError(f"Formato não suportado: {ext}")
    
    def _load_pdf(self, file_path: str) -> IngestedDocument:
        reader = PdfReader(file_path)

        pages_text = []
        full_text = []

        for page in reader.pages:
            text = page.extract_text() or ""
            text = self._normalize_text(text)

            pages_text.append(text)
            full_text.append(text)

        first_page = reader.pages[0]
        width = first_page.mediabox[2]
        height = first_page.mediabox[3]
        aspect_ratio = width / height

        if aspect_ratio > 1.1:
            doc_type = DocumentType.SLIDES
        else:
            doc_type = DocumentType.TEXT_PDF

        metadata = {
            "source": os.path.basename(file_path),
            "file_path": file_path,
            "num_pages": len(pages_text),
            "doc_type": doc_type.value
        }

        return IngestedDocument(
            text="\n".join(full_text),
            pages=pages_text,
            metadata=metadata
        )
    
    def _normalize_text(self, text: str) -> str:
        text = text.replace("\x00", "")
        text = text.strip()
        return text

# Parsers por tipo de documento
class BaseParser:
    def parse(self, doc: IngestedDocument) -> IngestedDocument:
        raise NotImplementedError
    
class PDFTextParser(BaseParser):
    def parse(self, doc: IngestedDocument) -> IngestedDocument:
        # Texto contínuo não tem estrutura confiável
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
            return PDFTextParser()

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

        doc_type = doc.metadata.get("doc_type")
        print(f"[INGESTION] Detected document type: {doc_type}")

        parser = ParserFactory.get_parser(doc_type) # type: ignore
        print(f"[INGESTION] Using parser: {parser.__class__.__name__}")
        doc = parser.parse(doc)

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
    chunks_text = pipeline.ingest(
        "pdfs/SAIA-Chapter12.pdf"
    )

    print(chunks_text[0].metadata)
    print(chunks_text[1].metadata)
    print(chunks_text[1].content[:200])

    # Slides
    chunks_slides = pipeline.ingest(
        "pdfs/SAIA-Chapter2-slide.pdf"
    )

    print(chunks_slides[0].metadata)
    print(chunks_slides[1].metadata)
    print(chunks_slides[1].content[:200])


    

if __name__ == "__main__":
    main()