#!/usr/bin/env python3

#################################################################################
# Loads raw documentation files into a summarized format that is prepared for RAG
#################################################################################

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
import os
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import START, StateGraph, END, MessagesState
from typing import List
from pydantic import BaseModel

# Prepare the LLM
load_dotenv()
llm = init_chat_model('us.anthropic.claude-3-5-haiku-20241022-v1:0', model_provider='bedrock_converse', temperature=0)


class ModuleState(MessagesState):
    doc: str
    name: str
    documents: List[str]


def glean_basics(state: ModuleState):
    class BasicsSchema(BaseModel):
        name: str
        version: str
        description: str
        author: str
        categories: List[str]
        parameter_names: List[str]

    # Define the prompt
    structured = llm.bind_tools([BasicsSchema])

    # Set the prompt
    prompt = ("Please extract the name, version, description, author, and categories from the GenePattern module",
              f"documentation below. If you do not know a value, leave it blank.\n\n{state['doc']}")

    # Invoke the model
    response = structured.invoke(prompt)
    structured_output = response.tool_calls[0]["args"]

    # Add basics documents
    state["documents"].append(f"{structured_output['name']} is on version {structured_output['version']}.")
    state["documents"].append(f"A description of {structured_output['name']} is {structured_output['description']}.")
    state["documents"].append(f"{structured_output['name']} is a GenePattern module written by {structured_output['author']}.")
    state["documents"].append(f"{structured_output['name']} can be categorized as {', '.join(structured_output['categories'])}.")
    state["documents"].append(f"{structured_output['name']} has the following parameters: {', '.join(structured_output['parameter_names'])}.")

    return {
        "name": structured_output["name"],
        "documents": state["documents"],
    }


def invoke_with_doc(state, prompt, rag_format=True):
    if rag_format: format_desc = """Format your description in embedding-friendly chunks for ingestion in a chroma 
        vector store. Break the content into atomic, semantically distinct chunks, with and natural language phrasing. 
        Write one chunk per line. Only write the text of the chunk; do not write metadata. Include the name of the 
        module somewhere in each chunk. Do not include any other text."""
    else: format_desc = ''

    # Get the response and append to messages
    response = llm.invoke(prompt + ' ' + format_desc + '\n\n' + state["doc"])
    state["messages"].append(response)

    # Extract documents
    for line in response.content.split('\n'):
        if line.strip(): state["documents"].append(line.strip())

    return state


def documentize(state: ModuleState):
    prompt = """Please give a technically detailed description the following GenePattern module documentation. It should 
    be targeted at someone with an undergraduate level of biological knowledge."""

    state = invoke_with_doc(state, prompt)
    return { "messages": state["messages"], "documents": state["documents"] }


def glean_uses(state: ModuleState):
    prompt = f"""Please describe the various uses of the {state["name"]} GenePattern module, both within the context of 
    GenePattern, as well as within the greater bioinformatics ecosystem. Be detailed and specific in your description. 
    It should be targeted at someone with an undergraduate level of biological knowledge. Use the knowledge you already 
    possess, as well as that found in the module documentation below."""

    state = invoke_with_doc(state, prompt)
    return {"messages": state["messages"], "documents": state["documents"]}


def glean_parameters(state: ModuleState):
    prompt = """Please describe each parameter detailed in the module documentation below, one parameter per line. You 
    should include the name of the parameter, its type, a description of what it does, and whether or not it is 
    required. If there are any default values, include those as well."""

    state = invoke_with_doc(state, prompt)
    return {"messages": state["messages"], "documents": state["documents"]}


def glean_formats(state: ModuleState):
    prompt = """Please describe the input and output files used by the GenePattern module in the documentation below. 
    Include the file format, contents and any other relevant information in your description. Describe one input or 
    output per line."""

    state = invoke_with_doc(state, prompt)
    return {"messages": state["messages"], "documents": state["documents"]}


def build_langgraph():
    """Build and compile the LangGraph for processing module information"""
    workflow = StateGraph(ModuleState)

    # Add new nodes
    workflow.add_node("glean_basics", glean_basics)
    workflow.add_node("glean_uses", glean_uses)
    workflow.add_node("glean_parameters", glean_parameters)
    workflow.add_node("glean_formats", glean_formats)
    workflow.add_node("documentize", documentize)

    # Define edges
    workflow.add_edge(START, "glean_basics")
    workflow.add_edge("glean_basics", "glean_uses")
    workflow.add_edge("glean_uses", "glean_parameters")
    workflow.add_edge("glean_parameters", "glean_formats")
    workflow.add_edge("glean_formats", "documentize")
    workflow.add_edge("documentize", END)

    # Compile the graph
    app = workflow.compile()
    return app


def load_pdf(file_path):
    print(f"Loading PDF at {file_path}")
    loader = PyPDFLoader(file_path)
    pages = []
    for page in loader.lazy_load(): pages.append(page)
    return ' '.join([page.page_content for page in pages])


def load_html(file_path):
    print(f"Loading HTML at {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        soup = BeautifulSoup(html_content, 'html.parser')
        text_content = soup.get_text(separator=' ', strip=True)
        return text_content
    except FileNotFoundError:
        raise f"File not found: {file_path}"
    except Exception as e:
        raise f"An error occurred while processing {file_path}: {e}"


def write_summary(directory, basename, content):
    """Write the given content to <basename>.txt in the specified directory."""
    if not os.path.exists(directory):
        os.makedirs(directory)  # Create the directory if it doesn't exist

    file_path = os.path.join(directory, f"{basename}.txt")
    print(f"Writing {file_path} to disk")
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)


def summarize_html(doc):
    initial_state = ModuleState(
        doc=doc,
        name="",
        messages=[],
        documents=[],
    )
    final_state = graph.invoke(initial_state)
    contents = "\n".join(final_state["documents"])
    return final_state.get('name'), contents


def summarize_all_doc(read_dir, write_dir):
    if not os.path.isdir(read_dir):
        raise ValueError(f"The path '{read_dir}' is not a valid directory.")

    for filename in os.listdir(read_dir):
        if os.path.exists(os.path.join(write_dir, os.path.basename(filename)[:-5] + '.txt')):
            print(f"Skipping {filename}")
            continue
        if filename.lower().endswith('.html') or filename.lower().endswith('.pdf'):
            full_path = os.path.join(read_dir, filename)
            if filename.lower().endswith('.pdf'):
                content = load_pdf(full_path)
            else: content = load_html(full_path)
            module_name, summary = summarize_html(content)
            write_summary(write_dir, module_name, summary)


# Prepare the summar0y graph
graph = build_langgraph()

# Summarize all HTML documentation files
summarize_all_doc('./library/moduledoc/raw', './library/moduledoc/')

# TEST WITH ONLY A SINGLE MODULE
# content = load_html('./library/moduledoc/raw/DESeq2.html')
# module_name, summary = summarize_html(content)
# write_summary('./library/moduledoc/', module_name, summary)

print("All raw files loaded into summarized library")
