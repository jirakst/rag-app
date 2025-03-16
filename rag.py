#!/usr/bin/env python
from typing import List
import os
import streamlit as st

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

# Remove dotenv loading since we'll use Streamlit secrets
# from dotenv import load_dotenv
# load_dotenv()

# 4. Streamlit app
st.title("LangChain RAG Demo")
st.write("This is a simple RAG demo using LangChain.")

# Add sidebar for API key input
with st.sidebar:
    st.header("Configuration")
    user_api_key = st.text_input("Enter your OpenAI API key", type="password")
    
# Determine which API key to use (user input takes precedence)
if user_api_key:
    openai_api_key = user_api_key
else:
    # Fallback to secrets or environment variable
    openai_api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))

# Check if we have an API key before proceeding
if not openai_api_key:
    st.warning("Please enter an OpenAI API key in the sidebar to use this app.")
    st.stop()

# 1. Load Retriever
# loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
loader = TextLoader("data/test.txt")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

# Use the determined API key
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
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
# Use the determined API key
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. Add a text input and a button
text_input = st.text_input("Enter a question")
button = st.button("Search")

if button:
    st.write(agent_executor.invoke({"input": text_input}))
