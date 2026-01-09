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

# ---------------------- CONFIGURAÇÕES ----------------------------
os.environ["USER_AGENT"] = "my-rag/1.0.0"

PDF_FOLDER = "pdfs/"
MODEL_NAME = "gpt-4o"
# gpt-4o-mini também é uma boa opção
EMBED_MODEL = "text-embedding-3-large"

class RAGState(TypedDict):
    messages: List[BaseMessage]
    rewrite_count: int

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
#print(chunks[0])

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

response_model = init_chat_model(MODEL_NAME, temperature=0)

def generate_query_or_respond(state: RAGState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retrieval tool or simply 
    respond to the user."""
    if state["rewrite_count"] >= 3:
        return {
            "messages": [
                HumanMessage(
                    content="This question is not related to the knowledge base topic."
                )
            ]
        }

    response = (
        response_model
        .bind_tools([retriever_tool]).invoke(state["messages"])  
    )
    return {"messages": [response]}

GRADE_PROMPT = (
    "You are an evaluator analyzing the relevance of a retrieved document to a user's question.\n"
    "Here is the retrieved document:\n\n {context} \n\n"
    "Here is the user's question: {question}\n"
    "If the document contains keywords or semantic meaning related to the user's question, classify it as relevant.\n"
    "Provide a binary score 'yes' or 'no' to indicate if the document is relevant to the question."
)

class GradeDocuments(BaseModel):
    """Evaluates documents using a binary score for relevance checking."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

grader_model = init_chat_model(MODEL_NAME, temperature=0)

def grade_documents(
    state: RAGState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Determines if the retrieved documents are relevant to the question."""
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = (
        grader_model
        .with_structured_output(GradeDocuments).invoke(  
            [{"role": "user", "content": prompt}]
        )
    )
    score = response.binary_score

    if score == "yes":
        return "generate_answer"
    
    if state["rewrite_count"] >= 3:
        return "generate_query_or_respond"
    
    return "rewrite_question"


REWRITE_PROMPT = (
    "Analyze the input and try to reason about the underlying intent or semantic meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved version of the question:"
)

def rewrite_question(state: RAGState):
    """Rewrites the original question."""
    messages = state["messages"]
    question = messages[0].content

    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])

    return {"messages": [HumanMessage(content=response.content)], "rewrite_count": state["rewrite_count"] + 1}



GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following retrieved context to answer the question. "
    #"If the provided context is insufficient or unrelated to the question, "
    #"respond exactly with: "
    #"'This question is not related to the available content.' "
    "Do not fabricate information. "
    "Do not try to infer or answer using external knowledge. "
    "Use no more than three sentences and keep the response concise.\n"
    "Question: {question}\n"
    "Context: {context}"
)


def generate_answer(state: RAGState):
    """Generates an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


# ---------------------- STATE GRAPH ----------------------

#workflow = StateGraph(MessagesState)
workflow = StateGraph(RAGState)

workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(rewrite_question)
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
workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")
graph = workflow.compile()


print("\n===== RAG Interativo =====")
print("Digite sua pergunta ou 'sair' para encerrar.\n")

while True:
    question = input("Pergunta: ").strip()

    if question.lower().strip() in ["sair", "exit", "quit"]:
        print("Encerrando...")
        break

    response = graph.invoke({"messages": [{"role": "user", "content": question}], "rewrite_count": 0})
    
    print("\n--- RESPOSTA ---\n")
    print(response["messages"][-1].content)
    print("\n----------------\n")