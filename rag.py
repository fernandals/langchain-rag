import os
import bs4
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=3, score_threshold=0.7)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# ---------------------- CRIAÇÃO DO AGENTE ----------------------------
prompt = (
    "You have access to a tool that retrieves context from the PDF documents. "
    "Always use the tool to gather information before answering the user."
)

agent = create_agent(model, [retrieve_context], system_prompt=prompt)


print("\n===== RAG Interativo =====")
print("Digite sua pergunta ou 'sair' para encerrar.\n")

while True:
    user_query = input("\nPergunta: ")

    if user_query.lower().strip() in ["sair", "exit", "quit"]:
        print("Encerrando...")
        break

    print("\n--- RESPOSTA ---\n")

    '''
    for event in agent.stream(
        {"messages": [{"role": "user", "content": user_query}]},
        stream_mode="values",
    ):
        event["messages"][-1].pretty_print()
    '''
    response = agent.invoke(
        {"messages": [{"role": "user", "content": user_query}]}
    )
    final_response = response["messages"][-1]
    final_response.pretty_print()

    print("\n----------------\n")