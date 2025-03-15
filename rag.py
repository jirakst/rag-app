#!/usr/bin/env python
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage

import streamlit as st

# 1. Load Retriever
# loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
loader = TextLoader("data/test.txt")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()

# 2. Create Tools
retriever_tool = create_retriever_tool(
    retriever,
    "john_doe_cv",
    "Search for information about John Doe's Curriculum Vitae (CV) including education, experience, skills, etc.",
)
# search = TavilySearchResults()
tools = [retriever_tool]


# 3. Create Agent
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# 4. Streamlit app
st.title("LangChain RAG Demo")
st.write("This is a simple RAG demo using LangChain.")

# 5. Add a text input and a button
text_input = st.text_input("Enter a question")
button = st.button("Search")

if button:
    st.write(agent_executor.invoke({"input": text_input}))
