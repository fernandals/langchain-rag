from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os

if __name__ == "__main__":
  discipline_name = "Software Architecture"
  persist_dir = f"data/knowledge_bases/{discipline_name}"

  embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBED_MODEL", "text-embedding-3-large")
  )

  vectorstore = Chroma(
    persist_directory=persist_dir,
    embedding_function=embeddings
  )

  retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

  query = "What is software architecture?"

  docs = retriever.invoke(query)

  if not docs:
    print("Nenhum documento encontrado.")

  for i, doc in enumerate(docs, 1):
    print(f"\nResultado {i}")
    print("-" * 50)

    print("Conteúdo (300 char):")
    print(doc.page_content[:300])

    print("\nMetadata:")
    for key, value in doc.metadata.items():
      print(f"{key}: {value}")
  