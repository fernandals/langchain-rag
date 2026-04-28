from pathlib import Path
from rag.loader import load_documents, parse_documents
from rag.splitter import spliting_documents
from rag.vectorstore import build_and_persist_vectorstore
from rag.models import KnowledgeBase
import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

def create_and_save_knowledge_base(folder_path: str, discipline_name: str):
  persist_dir = f"data/knowledge_bases/{discipline_name}"

  print("[LOAD] Carregando documentos da disciplina ...")
  raw_docs = load_documents(folder_path)

  print("[PARSE] Processando documentos ...")
  parsed_docs = parse_documents(raw_docs)

  print("[SPLIT] Dividindo documentos em chunks ...")
  chunks = spliting_documents(parsed_docs)

  print("[VECTORSTORE] Construindo base de conhecimento ...")
  vectorstore = build_and_persist_vectorstore(chunks, persist_dir)

  print(f"Base salva em: {persist_dir}")

  return KnowledgeBase(
      name=discipline_name,
      retriever=vectorstore.as_retriever()
  )

def load_knowledge_base(discipline_name: str):
  persist_dir = f"data/knowledge_bases/{discipline_name}"

  embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBED_MODEL", "text-embedding-3-large")
  )

  vectorstore = Chroma(
    persist_directory=persist_dir,
    embedding_function=embeddings
  )

  retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
  
  return retriever

#   return KnowledgeBase(
#       name=discipline_name,
#       retriever=vectorstore.as_retriever()
#   )
