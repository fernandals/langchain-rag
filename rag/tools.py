from langchain.tools import tool
from langchain_core.vectorstores import VectorStoreRetriever
from dotenv import load_dotenv
import os

load_dotenv()

retrieved_docs = int(os.getenv("RETRIEVED_DOCS", 3))

@tool
def retrieve_info(retriever: VectorStoreRetriever, query: str) -> str:
    """Search and return information about architectural styles in SysADL."""
    docs = retriever.invoke(query)[:retrieved_docs]
    return "\n\n".join([doc.page_content.strip() for doc in docs])
