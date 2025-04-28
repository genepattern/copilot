#!/usr/bin/env python3

#################################################################################
# Builds a new chroma vector store from the summarized library
#################################################################################

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set the embeddings model
print("Loading embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create the vector store
vector_store = Chroma(
    collection_name="moduledoc",
    embedding_function=embeddings,
    persist_directory="./chroma",
)

# Clear any existing documents in the vector store
print("Initializing the vector store...")
vector_store.reset_collection()

# Create the loader
loader = DirectoryLoader(
    "./library",      # replace with your directory
    glob="**/*.txt",
    loader_cls=UnstructuredMarkdownLoader
)

# Load the documents into the vector store
print("Loading documents...")
docs = loader.load()

# Split the documents into chunks
print("Splitting documents...")
splitter = RecursiveCharacterTextSplitter(separators=['\n'])
splits = splitter.split_documents(docs)

# Add the document chunks to the vector store
print("Ingesting documents...")
vector_store.add_documents(documents=splits)

# All done!
print("All files loaded and indexed in chroma")

