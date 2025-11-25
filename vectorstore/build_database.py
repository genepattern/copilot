#!/usr/bin/env python3

#################################################################################
# Builds a new chroma vector store from the summarized library
#################################################################################

import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path

# Set the embeddings model
print("Loading embeddings...")
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create the ChromaDB client
print("Initializing the vector store...")
chroma_client = chromadb.PersistentClient(path="./chroma")

# Delete the collection if it exists and create a new one
try:
    chroma_client.delete_collection(name="moduledoc")
    print("Deleted existing collection")
except:
    pass

vector_store = chroma_client.create_collection(
    name="moduledoc",
    embedding_function=embedding_function
)

# Load all text files from the library directory
print("Loading documents...")
documents = []
metadatas = []
ids = []

library_path = Path("./library")
txt_files = list(library_path.glob("**/*.txt"))

for idx, file_path in enumerate(txt_files):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by newlines to get individual chunks
    lines = [line.strip() for line in content.split('\n') if line.strip()]

    for line_idx, line in enumerate(lines):
        documents.append(line)
        metadatas.append({
            "source": str(file_path.relative_to(library_path)),
            "file_name": file_path.stem,
        })
        ids.append(f"{file_path.stem}_{idx}_{line_idx}")

# Add the document chunks to the vector store
print(f"Ingesting {len(documents)} document chunks...")
if documents:
    # ChromaDB has a limit on batch size, so we'll add in batches
    batch_size = 5000
    for i in range(0, len(documents), batch_size):
        end_idx = min(i + batch_size, len(documents))
        print(f"Adding batch {i // batch_size + 1} ({i + 1}-{end_idx} of {len(documents)})")
        vector_store.add(
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx]
        )

# All done!
print("All files loaded and indexed in chroma")
