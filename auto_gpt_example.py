from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
# Set up the memory
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
import faiss
# Set up the model and AutoGPT
from langchain.experimental import AutoGPT
from langchain.chat_models import ChatOpenAI
import os
import streamlit as st



st.title(f"Type a task here")
context = """
    You are an AI expert use the following url 
    https://dev.to/dunithd/how-to-structure-a-perfect-technical-tutorial-21h9
    https://realpython.com/practical-prompt-engineering/
    to learn how to write full and complete tutorials for beginners then, 
 """
user_input = st.text_input("", key="input")
prompt = context + user_input
while user_input != "":
    # Set up the tools
    search = GoogleSearchAPIWrapper()
    tools = [
        Tool(
            name="search",
            func=search.run,
            description="Useful for when you need to answer questions about tutorial. You should ask targeted questions",
            return_direct=True
        ),
        WriteFileTool(),
        ReadFileTool(),
    ]

    embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    embedding_size = 1536

    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    agent = AutoGPT.from_llm_and_tools(
        ai_name="Jim",
        ai_role="Assistant",
        tools=tools,
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
        memory=vectorstore.as_retriever()
    )

    # Set verbose to be true
    agent.chain.verbose = True

    response = agent.run([prompt])
    user_input = ""

    print(response)