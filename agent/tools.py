from langchain.tools import tool
from langchain_core.vectorstores import VectorStoreRetriever
from dotenv import load_dotenv
import os

load_dotenv()

retrieved_docs = int(os.getenv("RETRIEVED_DOCS", 3))

from langchain.tools import tool

def build_retrieve_tool(retriever: VectorStoreRetriever):

    @tool
    def retrieve(query: str) -> str:
        """Tool function to retrieve relevant information from the vector store based on the query."""
        docs = retriever.invoke(query)[:retrieved_docs]
        return "\n\n".join([doc.page_content.strip() for doc in docs])

    return retrieve