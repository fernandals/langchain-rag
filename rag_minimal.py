import os
import bs4
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import Literal
from langchain.messages import HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
from langchain_core.messages import BaseMessage
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from typing_extensions import Annotated
from langgraph.graph import add_messages
from langgraph.graph import MessagesState

# ---------------------- CONFIGURAÇÕES ----------------------------
os.environ["USER_AGENT"] = "my-rag/1.0.0"

PDF_FOLDER = "pdfs/"
MODEL_NAME = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-large"

class RAGState(MessagesState):
    documents: list[str]

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

docs_list = [doc for doc in docs]

print("Fazendo split dos documentos...")
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
chunks = text_splitter.split_documents(docs)
print(f"> Gerados {len(chunks)} chunks.")

vectorstore = InMemoryVectorStore.from_documents(
    documents=chunks, embedding=embeddings
)
retriever = vectorstore.as_retriever()

# ---------------------- TOOLS ----------------------

@tool
def retrieve_info(query: str) -> str:
    """Search and return information about architectural styles in SysADL."""
    docs = retriever.invoke(query)[:3]
    return "\n\n".join([doc.page_content.strip() for doc in docs])

retriever_tool = retrieve_info

SYSADL_SYSTEM_PROMPT = """
You are a domain-specific assistant specialized exclusively in SysADL architectural styles.

You are only allowed to handle questions that belong to the SysADL architectural styles domain.
If a user asks about anything outside this domain, you must refuse by saying:
"This question is not related to the available content."

Never answer questions about general knowledge, science, geography, history, or any topic
that is not strictly about SysADL architectural styles.
""".strip()

response_model = init_chat_model(
    MODEL_NAME, 
    temperature=0
).bind_tools([retriever_tool])

def generate_query_or_respond(state: RAGState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retrieval tool or simply 
    respond to the user."""

    response = (
        response_model.invoke([SystemMessage(SYSADL_SYSTEM_PROMPT)] + state["messages"])  
    )
    return {"messages": [response]}

GENERATE_PROMPT = """
You must decide whether the provided context explicitly contains the answer
to the given question about SysADL architectural styles.

Follow this procedure:

1. If the question is not about SysADL architectural styles, respond exactly:
"This question is not related to the available content."

2. If the question is about SysADL but the provided context does NOT explicitly
and directly contain the answer, respond exactly:
"This question is not related to the available content."

3. Only if the answer is clearly and directly supported by the context, you may answer.

Strict rules:
- Do NOT use external knowledge.
- Do NOT infer, assume, or complete missing information.
- Do NOT provide partial answers.
- The answer must be fully grounded in the provided context.
- Use no more than three sentences.

Question:
{question}

Context:
{context}
""".strip()

def generate_answer(state: RAGState):
    """Generates an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content

    system_prompt = SystemMessage(
        content=GENERATE_PROMPT.format(
            question=question,
            context=context
        )
    )
    
    response = response_model.invoke(
        [system_prompt] + state["messages"]
    )

    return {"messages": [response]}


# ---------------------- STATE GRAPH ----------------------

workflow = StateGraph(RAGState)

workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")
workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)
workflow.add_edge("retrieve", "generate_answer")
workflow.add_edge("generate_answer", END)
graph = workflow.compile()

print("--> Graph Visualization:")
print(graph.get_graph().draw_ascii())

print("\n===== RAG INTERATIVO (COM ETAPAS) =====")
print("Digite sua pergunta ou 'sair' para encerrar.\n")

while True:
    question = input("Pergunta: ").strip()

    if question.lower() in ["sair", "exit", "quit"]:
        print("Encerrando...")
        break

    print("\n===== INÍCIO DO PIPELINE RAG =====\n")

    # Estado inicial
    inputs = {
        "messages": [{"role": "user", "content": question}],
    }

    for step in graph.stream(inputs):
        for node_name, state in step.items():
            print(f"\n--- NÓ EXECUTADO: {node_name} ---")

            if "messages" in state:
                print("\nMensagens:")
                for msg in state["messages"]:
                    print(f"\nTipo: {type(msg).__name__}")

                    if isinstance(msg, HumanMessage):
                        print("Human:", msg.content)

                    elif isinstance(msg, AIMessage):
                        if msg.content:
                            print("AI:", msg.content)
                        if msg.tool_calls:
                            print("Tool calls:", msg.tool_calls)

                    elif isinstance(msg, ToolMessage):
                        print("Tool result:", msg.content)

    final_state = state
    print("\n===== RESPOSTA FINAL =====\n")
    print(final_state["messages"][-1].content)
    print("\n==========================\n")
