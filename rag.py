import os
from typing import TypedDict, Optional

import bs4
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, START, END

# ---------------------- CONFIGURAÇÕES ----------------------------
os.environ["USER_AGENT"] = "my-rag/1.0.0"

PDF_FOLDER = "pdfs/"
MODEL_NAME = "gpt-4"
EMBED_MODEL = "text-embedding-3-large"

print("Carregando modelo...")
model = ChatOpenAI(model=MODEL_NAME)
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

print(f"Carregando PDFs da pasta '{PDF_FOLDER}' ...")
loader = DirectoryLoader(
    PDF_FOLDER,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True
)
docs = loader.load()

print(f"> Total de páginas carregadas: {len(docs)}")
print(f"> Total de caracteres: {sum(len(doc.page_content) for doc in docs)}")

print("Fazendo split dos documentos...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
chunks = text_splitter.split_documents(docs)
print(f"> Gerados {len(chunks)} chunks.")

print("Criando vector store...")
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(chunks)

# ---------------------- STATE GRAPH ----------------------
class GraphState(TypedDict):
    question: str
    context: Optional[str]
    answer: Optional[str]


def retrieve(state: GraphState) -> GraphState:
    print("→ Buscando contexto...")
    docs = vector_store.similarity_search(state["question"], k=3, score_threshold=0.7)
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    return {
        "question": state["question"],
        "context": context,
        "answer": None,
    }

def generate(state: GraphState) -> GraphState:
    print("→ Gerando resposta...")

    context = state.get("context") or "Nenhum contexto encontrado."
    question = state["question"]

    prompt = f"""
Você é um assistente RAG. Use o contexto abaixo para responder.

Contexto:
{context}

Pergunta:
{question}

Resposta:"""

    result = model.invoke(prompt)

    return {
        "question": question,
        "context": context,
        "answer": result.content,
    }

# sempre vai pro retrieve para simplificar
def should_retrieve(state: GraphState) -> str:
    return "retrieve"


graph = StateGraph(GraphState)

graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

graph.add_conditional_edges(START, should_retrieve)

graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

executor = graph.compile()


print("\n===== RAG Interativo =====")
print("Digite sua pergunta ou 'sair' para encerrar.\n")

while True:
    question = input("Pergunta: ").strip()

    if question.lower().strip() in ["sair", "exit", "quit"]:
        print("Encerrando...")
        break

    print("\n--- PROCESSANDO... ---\n")

    state = {"question": question, "context": None, "answer": None}
    result = executor.invoke(state)

    print("\n--- RESPOSTA ---\n")
    print(result["answer"])
    print("\n----------------\n")