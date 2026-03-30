import os
import bs4
from utils import softmax
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
from dotenv import load_dotenv

load_dotenv()

# ---------------------- CONFIGURAÇÕES ----------------------------
os.environ["USER_AGENT"] = "my-rag/1.0.0"

PDF_FOLDER = "pdfs/"
MODEL_NAME = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-large"

class StudentProfile(BaseModel):
    asks_exercise: int = 0
    asks_detail: int = 0
    asks_objectivity: int = 0

    current_profile: str = "neutral"
    confidence: float = 0.0

class RAGState(MessagesState):
    documents: list[str]
    profile: StudentProfile

print("Carregando modelo...")
model = ChatOpenAI(model=MODEL_NAME)
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

print(f"Carregando PDFs da pasta '{PDF_FOLDER}' ...")
loader = DirectoryLoader(
    PDF_FOLDER,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader, # type: ignore
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

def update_profile(profile: StudentProfile, user_message: str) -> StudentProfile:
    DECAY = 0.8

    profile.asks_exercise *= DECAY
    profile.asks_detail *= DECAY
    profile.asks_objectivity *= DECAY
    
    text = user_message.lower()

    # heuristicas
    if "exerc" in text or "pratique" in text:
        profile.asks_exercise += 1

    if "detail" in text or "example" in text:
        profile.asks_detail += 1

    if "summarize" in text or "direct" in text:
        profile.asks_objectivity += 1

    scores = {
        "analytical": profile.asks_detail,
        "explorer": profile.asks_exercise,
        "objective": profile.asks_objectivity,
    }

    probs = softmax(scores)

    # profile_score = max(scores, key=scores.get)
    # max_score = scores[profile_score]

    # total = sum(scores.values())

    # if total > 0:
    #     profile.current_profile = profile_score
    #     profile.confidence = max_score / total

    profile.current_profile = max(probs, key=probs.get)
    profile.confidence = probs[profile.current_profile]

    return profile

# ---------------------- TOOLS ----------------------

@tool
def retrieve_info(query: str) -> str:
    """Search and return information about architectural styles in SysADL."""
    docs = retriever.invoke(query)[:3]
    return "\n\n".join([doc.page_content.strip() for doc in docs])

retriever_tool = retrieve_info

SYSADL_SYSTEM_PROMPT = """
You are an educational assistant acting as an intelligent monitor for a course.

IMPORTANT:
When a user asks a question about SysADL,
you MUST call the retrieval tool to inspect the course material
before deciding whether the question is related.

Your role is to help students understand SysADL architectural styles,
but you must NEVER provide direct answers, final definitions, or complete solutions.

You are specialized exclusively in SysADL architectural styles.
You are NOT allowed to answer questions outside this domain.

If a question is not strictly related to SysADL architectural styles,
respond exactly:
"This question is not related to the available content."

Pedagogical rules you must always follow:
- Do NOT give direct answers, definitions, or conclusions.
- Do NOT solve exercises or provide final results.
- Always guide the student through hints, questions, or reasoning steps.
- Encourage the student to think, reflect, and derive the answer themselves.
- You may reference concepts, relationships, or sections of the content,
  but never restate the answer explicitly.

You must behave as a tutor, not as an answer generator.
""".strip()

def build_dynamic_prompt(profile: StudentProfile) -> str:
    base = SYSADL_SYSTEM_PROMPT

    style_instruction = ""

    if profile.current_profile == "analytical":
        style_instruction = """
Additionally:
- Provide more detailed reasoning steps.
- Break explanations into structured logical parts.
- When possible, guide using practical examples or applied scenarios.
"""

    elif profile.current_profile == "explorer":
        style_instruction = """
Additionally:
- Prefer proposing small reflective exercises instead of extended explanations.
- Generate guiding questions that encourage exploration and self-discovery.
"""

    elif profile.current_profile == "objective":
        style_instruction = """
Additionally:
- Keep responses concise.
- Use short guiding questions instead of long explanations.
"""

    return base + "\n" + style_instruction

response_model = init_chat_model(
    MODEL_NAME, 
    temperature=0
).bind_tools([retriever_tool])

def generate_query_or_respond(state: RAGState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retrieval tool or simply 
    respond to the user."""

    last_user_msg = state["messages"][-1].content
    profile = state.get("profile", StudentProfile())
    profile = update_profile(profile, last_user_msg)

    state["profile"] = profile

    dynamic_system_prompt = build_dynamic_prompt(profile)

    response = (
        response_model.invoke(
            [SystemMessage(dynamic_system_prompt)] + state["messages"]
        )
    )

    return {"messages": [response], "profile": profile}

GENERATE_PROMPT = """
You must decide whether the provided context allows you to HELP the student
understand a question about SysADL architectural styles, without giving
the direct answer.

Follow this procedure:

1. If the question is NOT about SysADL architectural styles, respond exactly:
"This question is not related to the available content."

2. If the question is about SysADL but the provided context does NOT contain
information that can help guide the student, respond exactly:
"This question is not related to the available content."

3. If the context contains relevant information:
- Do NOT provide the answer directly.
- Do NOT restate definitions verbatim.
- Provide hints, guiding questions, or partial reasoning steps.
- Point the student to relevant concepts or relationships in the context.
- Encourage the student to formulate the answer themselves.

Strict rules:
- Do NOT use external knowledge.
- Do NOT infer beyond the context.
- Do NOT provide final answers or conclusions.
- Your response must be pedagogical and exploratory.
- Use at most three sentences.
- Never mention retrieving information or tools.
- Never ask the user if they want you to search for information.
- Continue the pedagogical interaction based on the student's last message.
- Stay focused on the specific architectural style mentioned in the question.

Question:
{question}

Context:
{context}
""".strip()

def generate_answer(state: RAGState):
    """Generates an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    profile = state.get("profile", StudentProfile())

    style = f"""
Student profile:
- Current profile: {profile.current_profile}
- Confidence: {profile.confidence:.2f}

Adapt the pedagogical style accordingly.
"""

    system_prompt = SystemMessage(
        content=GENERATE_PROMPT.format(
            question=question,
            context=context
        ) + "\n" + style
    )
    
    response = response_model.invoke(
        [system_prompt] + state["messages"]
    )

    return {"messages": [response], "profile": profile}


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

#student_profile = StudentProfile()

conversation_state = {
    "messages": [],
    "profile": StudentProfile()
}

while True:
    question = input("Pergunta: ").strip()

    if question.lower() in ["sair", "exit", "quit"]:
        print("Encerrando...")
        break

    print("\n===== INÍCIO DO PIPELINE RAG =====\n")

    # Estado inicial
    # inputs = {
    #     "messages": [{"role": "user", "content": question}],
    #     "profile": student_profile
    # }

    conversation_state["messages"].append(
        HumanMessage(content=question)
    )

    for step in graph.stream(conversation_state): # type: ignore
        for node_name, state in step.items():
            print(f"\n--- NÓ EXECUTADO: {node_name} ---")

            # DEBUG DO PERFIL
            if "profile" in state:
                profile = state["profile"]

                print("\n[DEBUG] Perfil do aluno:")
                print(f"perfil_atual: {profile.current_profile}")
                print(f"confianca: {profile.confidence:.2f}")
                print("sinais:")
                print(f"  pede_exercicio: {profile.asks_exercise}")
                print(f"  pede_detalhe: {profile.asks_detail}")
                print(f"  pede_objetividade: {profile.asks_objectivity}")

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

    conversation_state = state

    # final_state = state
    # student_profile = final_state.get("profile", StudentProfile())

    print("\n===== RESPOSTA FINAL =====\n")
    print(conversation_state["messages"][-1].content)
    print("\n===== PERFIL DO ALUNO =====\n")
    print(f"Perfil atual: {conversation_state['profile'].current_profile}")
    print(f"Confiança: {conversation_state['profile'].confidence:.2f}")
    print("\n==========================\n")
