from langchain.schema import Document

## ollama 
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


## chat history stuff
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import trim_messages


## lang graph
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.graph import START, MessagesState, StateGraph

## visualizing lang graph
# from IPython.display import Image, display
# from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles


## document loaders & text splitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.parsers import BS4HTMLParser
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_text_splitters import HTMLSectionSplitter
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community import embeddings as emb
from langchain_experimental.llms.ollama_functions import OllamaFunctions




## requests
import requests
from bs4 import BeautifulSoup


## chroma vector store
from langchain_chroma import Chroma
import chromadb
from langchain.retrievers import MultiQueryRetriever

## others
import pandas as pd
from tqdm import tqdm
from uuid import uuid4


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    ConfigurableField,
    RunnablePassthrough,
)

from langchain_aws import ChatBedrockConverse




## memory stuff and langgraph
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

