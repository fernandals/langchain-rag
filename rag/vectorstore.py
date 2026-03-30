import os
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from dotenv import load_dotenv

load_dotenv()

def build_vectorstore(chunks: list[list[Document]]) -> list[Document]:
  embeddings_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
  embeddings = OpenAIEmbeddings(model=embeddings_model)

  vectorstore = InMemoryVectorStore.from_documents(
    documents=[chunk for chunk_list in chunks for chunk in chunk_list], embedding=embeddings
  )

  return vectorstore.as_retriever()