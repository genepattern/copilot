#!/usr/bin/env python3

#################################################################################
# Loads raw documentation files into a summarized format that is prepared for RAG
#################################################################################

from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
from typing import List
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockConverseModel
from pypdf import PdfReader

# Prepare the LLM
load_dotenv()
model = BedrockConverseModel('us.anthropic.claude-3-5-haiku-20241022-v1:0')


class ModuleState:
    def __init__(self, doc: str, name: str = "", documents: List[str] = None):
        self.doc = doc
        self.name = name
        self.documents = documents if documents is not None else []
        self.messages = []


class BasicsSchema(BaseModel):
    name: str
    version: str
    description: str
    author: str
    categories: List[str]
    parameter_names: List[str]


def glean_basics(state: ModuleState):
    # Create an agent with structured output
    agent = Agent(model, output_type=BasicsSchema)

    # Set the prompt
    prompt = f"""Please extract the name, version, description, author, and categories from the GenePattern module
documentation below. If you do not know a value, leave it blank.

{state.doc}"""

    # Invoke the model
    result = agent.run_sync(prompt)
    structured_output = result.output

    # Add basics documents
    state.documents.append(f"{structured_output.name} is on version {structured_output.version}.")
    state.documents.append(f"A description of {structured_output.name} is {structured_output.description}.")
    state.documents.append(f"{structured_output.name} is a GenePattern module written by {structured_output.author}.")
    state.documents.append(f"{structured_output.name} can be categorized as {', '.join(structured_output.categories)}.")
    state.documents.append(f"{structured_output.name} has the following parameters: {', '.join(structured_output.parameter_names)}.")

    state.name = structured_output.name
    return state


def invoke_for_module(state, prompt, rag_format=True):
    if rag_format:
        format_desc = """Format your description in embedding-friendly chunks for ingestion in a chroma 
        vector store. Break the content into atomic, semantically distinct chunks, with and natural language phrasing. 
        Write one chunk per line. Only write the text of the chunk; do not write metadata. Include the name of the 
        module somewhere in each chunk. Do not include any other text."""
    else:
        format_desc = ''

    # Create an agent for text generation
    agent = Agent(model, output_type=str)

    # Get the response
    full_prompt = prompt + ' ' + format_desc + '\n\n' + state.doc
    result = agent.run_sync(full_prompt)
    response_content = result.output

    # Extract documents
    for line in response_content.split('\n'):
        if line.strip():
            state.documents.append(line.strip())

    return state


def invoke_for_doc(state, prompt, rag_format=True):
    if rag_format:
        format_desc = """Format your description in embedding-friendly chunks for ingestion in a chroma 
        vector store. Break the content into atomic, semantically distinct chunks, with and natural language phrasing. 
        Write one chunk per line. Only write the text of the chunk; do not write metadata. Do not include any other text."""
    else:
        format_desc = ''

    # Create an agent for text generation
    agent = Agent(model, output_type=str)

    # Get the response
    full_prompt = prompt + ' ' + format_desc + '\n\n' + state.doc
    result = agent.run_sync(full_prompt)
    response_content = result.output

    # Extract documents
    for line in response_content.split('\n'):
        if line.strip():
            state.documents.append(line.strip())

    return state


def module_documentize(state: ModuleState):
    prompt = """Please give a technically detailed description the following GenePattern module documentation. It should 
    be targeted at someone with an undergraduate level of biological knowledge."""

    state = invoke_for_module(state, prompt)
    return state


def server_documentize(state: ModuleState):
    prompt = """Please give a technically detailed description the following GenePattern documentation. It should 
    be targeted at someone with an undergraduate level of biological knowledge."""

    state = invoke_for_doc(state, prompt)
    return state


def glean_uses(state: ModuleState):
    prompt = f"""Please describe the various uses of the {state.name} GenePattern module, both within the context of 
    GenePattern, as well as within the greater bioinformatics ecosystem. Be detailed and specific in your description. 
    It should be targeted at someone with an undergraduate level of biological knowledge. Use the knowledge you already 
    possess, as well as that found in the module documentation below."""

    state = invoke_for_module(state, prompt)
    return state


def glean_parameters(state: ModuleState):
    prompt = """Please describe each parameter detailed in the module documentation below, one parameter per line. You 
    should include the name of the parameter, its type, a description of what it does, and whether or not it is 
    required. If there are any default values, include those as well."""

    state = invoke_for_module(state, prompt)
    return state


def glean_formats(state: ModuleState):
    prompt = """Please describe the input and output files used by the GenePattern module in the documentation below. 
    Include the file format, contents and any other relevant information in your description. Describe one input or 
    output per line."""

    state = invoke_for_module(state, prompt)
    return state


def process_module_doc(doc: str) -> tuple[str, str]:
    """Process module documentation through a sequential pipeline"""
    state = ModuleState(doc=doc)

    # Run the pipeline sequentially
    state = glean_basics(state)
    state = glean_uses(state)
    state = glean_parameters(state)
    state = glean_formats(state)
    state = module_documentize(state)

    contents = "\n".join(state.documents)
    return state.name, contents


def process_server_doc(doc: str) -> tuple[str, str]:
    """Process server documentation"""
    state = ModuleState(doc=doc)

    # Run the pipeline
    state = server_documentize(state)

    contents = "\n".join(state.documents)
    return state.name, contents


def load_pdf(file_path):
    print(f"Loading PDF at {file_path}")
    text_content = []

    with open(file_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text_content.append(page.extract_text())

    return ' '.join(text_content)


def load_html(file_path):
    print(f"Loading HTML at {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        soup = BeautifulSoup(html_content, 'html.parser')
        text_content = soup.get_text(separator=' ', strip=True)
        return text_content
    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}")
    except Exception as e:
        raise ValueError(f"An error occurred while processing {file_path}: {e}")


def write_summary(directory, basename, content):
    """Write the given content to <basename>.txt in the specified directory."""
    if not os.path.exists(directory):
        os.makedirs(directory)  # Create the directory if it doesn't exist

    file_path = os.path.join(directory, f"{basename}.txt")
    print(f"Writing {file_path} to disk")
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)


def summarize_html(doc, doc_type='module'):
    if doc_type == 'module':
        return process_module_doc(doc)
    else:
        return process_server_doc(doc)


def summarize_all_doc(doc_type, read_dir, write_dir):
    if not os.path.isdir(read_dir):
        raise ValueError(f"The path '{read_dir}' is not a valid directory.")

    for filename in os.listdir(read_dir):
        base_name = os.path.basename(filename)[:-5]
        if os.path.exists(os.path.join(write_dir, base_name + '.txt')):
            print(f"Skipping {filename}")
            continue
        if filename.lower().endswith('.html') or filename.lower().endswith('.pdf'):
            full_path = os.path.join(read_dir, filename)
            if filename.lower().endswith('.pdf'):
                content = load_pdf(full_path)
            else:
                content = load_html(full_path)
            page_name, summary = summarize_html(content, doc_type)
            if not page_name:
                page_name = base_name
            write_summary(write_dir, page_name, summary)


# Summarize all HTML documentation files
summarize_all_doc('module', './library/moduledoc/raw', './library/moduledoc/')
summarize_all_doc('server', './library/serverdoc/raw', './library/serverdoc/')
summarize_all_doc('server', './library/notebookdoc/raw', './library/notebookdoc/')

# TEST WITH ONLY A SINGLE MODULE
# content = load_html('./library/moduledoc/raw/DESeq2.html')
# module_name, summary = summarize_html(content, 'module')
# write_summary('./library/moduledoc/', module_name, summary)

print("All raw files loaded into summarized library")
