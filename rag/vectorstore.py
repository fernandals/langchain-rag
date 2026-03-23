import os
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from dotenv import load_dotenv

load_dotenv()

def build_vectorstore(chunks: list[Document]) -> list[Document]:
  embeddings_model = os.getenv("EMBED_MODEL")
  embeddings = OpenAIEmbeddings(model=embeddings_model)

  vectorstore = InMemoryVectorStore.from_documents(
    documents=chunks, embedding=embeddings
  )

  return vectorstore.as_retriever()