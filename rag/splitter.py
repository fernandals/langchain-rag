from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def spliting_documents(docs: list[Document]) -> list[Document]:
  print("Fazendo split dos documentos...")
  text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True  
  )

  chunks = text_splitter.split_documents(docs)
  print(f"> Gerados {len(chunks)} chunks.")

  return chunks