from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

def load_documents(folder_pth: Path) -> list[Document]:
  print(f"Carregando PDFs da pasta '{folder_pth}' ...")
  loader = DirectoryLoader(
    folder_pth,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader, # type: ignore
    show_progress=True
  )
  docs = loader.load()

  print(f"> Total de páginas carregadas: {len(docs)}")
  print(f"> Total de caracteres: {sum(len(doc.page_content) for doc in docs)}")

  docs_list = [doc for doc in docs]
  return docs_list
