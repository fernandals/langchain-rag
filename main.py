from rag.splitter import spliting_documents
from rag.vectorstore import build_vectorstore
from rag.tools import retrieve_info
import agent.prompts as prompts
from agent.graph import build_graph
from student_model.profile import StudentProfile
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

from pathlib import Path
from rag.loader import load_documents, parse_documents

if __name__ == "__main__":
  
  docs = load_documents(Path("pdfs/"))

  parsed_docs = parse_documents(docs)

  chunks_table = spliting_documents(docs)
  
  retriever = build_vectorstore(chunks_table)
