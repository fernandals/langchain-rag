import os
import streamlit as st
from typing import TypedDict, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

from langgraph.graph import StateGraph, START, END

# ---------------------- CONFIGURAÇÕES ----------------------------
os.environ["USER_AGENT"] = "my-rag-ui/1.0.0"

PDF_FOLDER = "pdfs/"
MODEL_NAME = "gpt-4"
EMBED_MODEL = "text-embedding-3-large"

@st.cache_resource(show_spinner=True)
def load_pipeline():

    model = ChatOpenAI(model=MODEL_NAME)
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

    loader = DirectoryLoader(
        PDF_FOLDER,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(docs)

    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(chunks)

    class GraphState(TypedDict):
        question: str
        context: Optional[str]
        answer: Optional[str]

    def retrieve(state: GraphState) -> GraphState:
        docs = vector_store.similarity_search(state["question"], k=3)
        context = "\n\n".join([d.page_content for d in docs])
        return {"question": state["question"], "context": context, "answer": None}

    def generate(state: GraphState) -> GraphState:
        context = state.get("context") or "Nenhum contexto encontrado"
        question = state["question"]

        prompt = f"""
Você é um assistente RAG. Use o contexto abaixo para responder:

Contexto:
{context}

Pergunta:
{question}

Resposta:
"""
        result = model.invoke(prompt)
        return {"question": question, "context": context, "answer": result.content}

    def should_retrieve(state: GraphState) -> str:
        return "retrieve"

    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_conditional_edges(START, should_retrieve)
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    executor = graph.compile()

    return executor


# ---------------------- STREAMLIT UI ----------------------------
st.set_page_config(page_title="RAG Assistente", layout="wide")

st.title("Assistente RAG (LangGraph + Streamlit)")

executor = load_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Digite sua pergunta...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Gerando resposta..."):
            state = {"question": user_query, "context": None, "answer": None}
            result = executor.invoke(state)
            answer = result["answer"]

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
