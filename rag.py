import os

os.environ["USER_AGENT"] = "my-rag/1.0.0"

import bs4

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool
from langchain.agents import create_agent

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# chat model
model = ChatOpenAI(model="gpt-4")

# embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# vector store
vector_store = InMemoryVectorStore(embeddings)

# selecting tags to keep from html page
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))

# loading document from url
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

# testing
assert len(docs) == 1
print(f"Total characters: {len(docs[0].page_content)}")
#print(docs[0].page_content[:500])

# spliting documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")

# storing documents
document_ids = vector_store.add_documents(documents=all_splits)

# setting the vector store to be a tool
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=3, score_threshold=0.7)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# creating agent
tools = [retrieve_context]
prompt = (
    "You have access to a tool that retrieves context from a blog post. "
    "Use the tool to help answer user queries."
)
agent = create_agent(model, tools, system_prompt=prompt)

query = (
    "What is the standard method for Task Decomposition?\n\n"
    "Once you get the answer, look up common extensions of that method."
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()

