from ..langchain_utilities import *
import pandas as pd
import os
from langchain_core.documents import Document
import json
import requests
import numpy as np
from .llama import *
from .langgraph_testing import *
from langchain_community.document_loaders import PythonLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.document_loaders import BSHTMLLoader
from langchain_core.documents.base import Blob
from langchain_community.document_loaders import PyMuPDFLoader
from tqdm import tqdm
from uuid import uuid4
from langgraph.prebuilt import create_react_agent


## this file contains scripts that creates different LLMs that help with summarizing documents...
## ... and loading them into their respective vector stores. 



def documentation_agent(document):
    """
    Returns an LLM chain for processing PDF documents.
    """
    model = get_model('llama-3.3')
    system_prompt = ChatPromptTemplate.from_template("""
    You are a document ingestion agent specialized in processing documents for a bioinformatics support system.
    
    The document might be a PDF or HTML. 

    **Preprocessing:**
    1. **Extract text:** Convert the document to plain text while preserving section headers.
    2. **Remove noise:** Strip out noisy headers, footers, page numbers, and OCR artifacts.
    3. **Normalize content:** Clean duplicate or boilerplate text and normalize whitespace.
    4. **Structure extraction:** Identify key sections (e.g., Abstract, Methods, Results, Conclusion).

    **Output:**

    ### **Section 1: Summary**
    Provide a human-readable summary that covers:
    - Name of the module
    - If this is part of a bigger pipeline, give the name of the main pipeline. 
        - For example, Seurat.Clustering is part of a bigger pipeline because Seurat is the main pipeline and Clustering is one of the steps. Notice the period in between Seurat and Clustering. 
    - Analysis methodology (e.g., RNA-seq, clustering, variant calling).
    - Intended use case and biological problem.
    - Typical downstream or companion analyses.
    - Biological context (species, disease, tissue, cell lines).
    - Tools, methods, datasets, or file formats mentioned.

    ### **Section 2: Structured Annotations**
    List the following elements:
    - `analysis_type`: One-line description (e.g., "Differential Gene Expression - RNA-seq")
    - `use_case`: The purpose of the analysis/tool.
    - `related_analyses`: Common follow-up analyses.
    - `biological_context`: Organism, disease, or experiment type.
    - `tools_or_modules`: Named tools, modules, or workflows mentioned.
    - `file_formats`: Input/output formats (e.g., GCT, FASTQ, BAM, VCF).
    - `parameters`: Parameter names with default values or descriptions.
    
    ### **Section 3: Potential questions:
    Give some potential questions that the users might ask, which will retrieve this document. 
    
    Here are some metadata about the document: 
    {metadata}
    
    Here is the document content:

    {document}
    """)

    chain = (
         {"document": RunnablePassthrough(), 
          'metadata' : RunnablePassthrough()}
         | system_prompt
         | model
         | StrOutputParser()
    )
    return chain.invoke({'document': document.page_content, 
                         'metadata': document.metadata})


def manifest_agent(document):
    """
    Returns an LLM chain for processing code snippets or scripts.
    """
    model = get_model('llama-3.3')
    system_prompt_code = ChatPromptTemplate.from_template("""
    You are a document ingestion agent specialized in processing code snippets 
    or scripts for a bioinformatics support system.
    
    **Preprocessing:**
    1. **Extract relevant code:** Isolate functional code blocks, inline comments, and documentation strings.
    2. **Remove extraneous text:** Strip out any non-functional boilerplate or unrelated text.
    3. **Identify all key elements:** Highlight function definitions, modules, dependencies, and any parameter settings.
    4. **Normalize formatting:** Maintain code formatting for readability and context.
    
    **Output:**
    
    ### **Section 1: Summary**
    Summarize:
    - The name of the module.
    - If this is part of a bigger pipeline, give the name of the main pipeline. 
        - For example, Seurat.Clustering is part of a bigger pipeline because Seurat is the main pipeline and Clustering is one of the steps. Notice the period in between Seurat and Clustering. 
    - The purpose and functionality of the code.
    - The main bioinformatics analysis or computational method implemented.
    - The intended use case and any biological context (if applicable).
    - Possible downstream analysis if applicable. 
    
    ### **Section 2: Structured Annotations**
    List these elements where applicable:
    - `analysis_type` (if relevant)
    - `use_case`
    - `related_analyses`
    - `biological_context` (if applicable)
    - `tools_or_modules`
    - `parameters` (all module parameters as specified in the manifest file, including configuration settings or variable defaults, and file formats.)
        - java.io.File is a File, specify the format if this exists, return "Takes in input file" for this
        - java.lang.String is just a string 
        - java.lang.Integer is an integer
        
    ### **Section 3: Potential questions:
    Give some potential questions that the users might ask, which will retrieve this document. The questions could be a bit more technical, such as asking about file formats and inputs. 
        
    Here is some metadata on the document:
    {metadata} 
    
    
    Here is the code content:
    {document}
    
    
    """)
    chain = (
         {"document": RunnablePassthrough(),
          "metadata": RunnablePassthrough()}
         | system_prompt_code
         | model
         | StrOutputParser()
    )
    return chain.invoke({'document': document.page_content, 'metadata' : document.metadata})


def guides_agent(document):
    """
    Returns an LLM chain for processing research articles or technical reports.
    """
    model = get_model('llama-3.3')
    system_prompt_research = ChatPromptTemplate.from_template("""
        You are a document ingestion agent specialized in processing GenePattern user guides and technical documentation for a bioinformatics support system.

        **Preprocessing:**
        1. **Section extraction:** Identify and separate key sections (Overview, Task Description, Required Inputs, Steps, Expected Output, Additional Notes).
        2. **Clean redundant content:** Remove non-essential parts such as repeated headers/footers and citation clutter.
        3. **Normalize text:** Ensure consistent formatting, including proper paragraph breaks and heading structures.
        4. **Preserve instructional content:** Extract figures, task steps, tables, and code snippets if they contribute to the guide's instructions.

        **Output:**

        ### **Summary** ###
        Provide a comprehensive summary that includes:
        - Include the header of the guide, along with the subsection. 
        - The main topic, objectives, and purpose of the user guide.
        - What specific task or analysis is being performed in GenePattern.
        - Step-by-step instructions or procedures mentioned in the document.
        
        Provide a list of questions the user might ask, to help retrieve this document. 
        
        Here are some metadata: 
        {metadata}

        Here is the document content:
        {document}
        
        """)
    chain = (
         {"document": RunnablePassthrough(),
          "metadata": RunnablePassthrough()}
         | system_prompt_research
         | model
         | StrOutputParser()
    )
    return chain.invoke({'document': document.page_content, 'metadata' : document.metadata})


extensions = {
    "py": "python",
    "js": "js",
    "cobol": "cobol",
    "c": "c",
    "cpp": "cpp",
    "cs": "csharp",
    "rb": "ruby",
    "scala": "scala",
    "rs": "rust",
    "go": "go",
    "kt": "kotlin",
    "lua": "lua",
    "pl": "perl",
    "ts": "ts",
    "java": "java",
    "php": "php",
    "ex": "elixir",
    "exs": "elixir",
    "sql": "sql",
}

def load_document2(fp):
    """
    Loads a document (more advanced) using "LanguageParser" for code, and pdf loaders
    """
    name, extension = os.path.splitext(fp)
    # print(f'Name of module: {name}, extension : {extension}')
    if extension.replace('.', '') in extensions:
        # print(f"parsing {extension.replace('.', '')} file")
        blob = Blob.from_path(fp)
        try:
            raw_documents = LanguageParser(language = extensions[extension.replace('.', '')]).parse(blob)
            for doc in raw_documents:
                doc.page_content = f"This function or file is the script for this module: {name}" + doc.page_content
                doc.metadata['documentation_type'] = 'script'
            docs = raw_documents
        except:
            raw_documents = TextLoader(fp).load()
            raw_pages = [raw_documents[i].page_content for i in range(len(raw_documents))]
            text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap = 40, length_function  = len)
            docs = text_splitter.create_documents(raw_pages)
    else:
        if ".pdf" == extension:
            raw_documents = PyMuPDFLoader(file_path=fp, extract_images = False, extract_tables = 'markdown').load()
            docs = raw_documents
        elif ".html" == extension:
            raw_documents = html.UnstructuredHTMLLoader(fp).load()
            docs = raw_documents
        elif '.Rmd' == extension:
            raw_documents = rmd_loader(fp)
            return raw_documents
        elif '.R' == extension:
            raw_documents = R_loader(fp)
            return raw_documents
        else:
            raw_documents = TextLoader(fp).load()
            raw_pages = [raw_documents[i].page_content for i in range(len(raw_documents))]
            text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap = 40, length_function  = len)
            docs = text_splitter.create_documents(raw_pages)

    for doc in docs:
        doc.metadata['source'] = fp
        doc.metadata['file_name'] = os.path.basename(fp)
        doc.metadata['module_name'] = fp.split('/')[-2]
        doc.metadata['document_type'] = extension
    return docs

def load_documents2(fps):
    """
    Loads multiple document and generates a list of documents for langchain 
    """
    docs = []
    for path in fps:
        try:
            fp = os.path.join(path)
            docs = docs + load_document2(fp)
            # os.system(f'rm {fp}')
        except Exception as e:
            # os.system(f'rm {fp}')
            continue
            # print(f'Unsupported file type {path}')
    return docs

def metadata_formatter(metadata_dct):
    """
    Formats metadata into string for LLMs to read
    """
    string = ""
    for k, v in metadata_dct.items():
        string += f'Here are some metadata about this document: \n The {k} is {v} \n '

    return string




def obtain_raw_documents(manifests, documentations, wrappers, threads, readmes, modules_that_exist):
    """
    A function to obtain all raw documents
    """
    
    ## start loading documents:
    ## manifests
    manifest_docs = []
    manifest_fps = [os.path.join(manifests, fp) for fp in os.listdir(manifests)]
    ## create metadata: manifest:
    for man_fp in manifest_fps:
        name = os.path.basename(man_fp).replace('.txt', '')
        if name in modules_that_exist:
            name2, extension = os.path.splitext(man_fp)
            metadata = {'module_name' : name,
                    'documentation_type' : 'manifest', 
                    'filename' : man_fp, 
                    'source' : 'github repos', 
                    'format' : extension}
                
            page_content = open(man_fp).read()
            manifest_docs.append(Document(metadata = metadata, page_content=page_content))
    print(f'There are a total of {len(manifest_docs)} manifests.')

    
    readme_docs = []
    readme_fps = [os.path.join(readmes, fp) for fp in os.listdir(readmes)]
    for readme_fp in readme_fps:
        name = os.path.basename(readme_fp).replace('.md', '')
        if name.replace('.README', '') in modules_that_exist:
            name2, extension = os.path.splitext(readme_fp)
            metadata = {'module_name' : name, 
                        'documentation_type' : 'readme', 
                    'filename':readme_fp,
                    'source': 'github repos',
                    'format' : extension}
            try:
                page_content = open(readme_fp).read()
                readme_docs.append(Document(
                    metadata = metadata,
                    page_content = page_content
                ))
            except:
                continue
    print(f'There are a total of {len(readme_docs)} read mes.')


    ## gp help 
    thread_docs = []
    original_questions = []
    answers = []
    thread_fps = [os.path.join(threads, fp) for fp in os.listdir(threads)]
    for thread_fp in thread_fps:
        name = os.path.basename(readme_fp).replace('.txt', '')
        name, extension = os.path.splitext(thread_fp)
        metadata = {'module_name' : name, 
                    'documentation_type' : 'GP help forum thread', 
                'filename':thread_fp,
                'source': 'GP help forum',
                'format' : extension}
        page_content = open(thread_fp).read()
        thread_docs.append(Document(
            metadata = metadata,
            page_content = page_content
        ))
        spl = page_content.split('---- NEW MESSAGE ----')
        if len(spl) < 5:
            ## only append the q and a from a 2 q 2 a thread. 
            original_questions.append(spl[1])
            answers.append(spl[-1])        

    print(f'There are a total of {len(thread_docs)} threads. ')



    ## wrappers
    # Collect all file paths
    wrapper_paths = []
    for root, _, files in os.walk(wrappers):
        module_name = os.path.basename(root)
        if module_name in modules_that_exist:
            for file in files:
                full_path = os.path.join(root, file)
                wrapper_paths.append(full_path)

    # Print or use the full file paths
    wrappers = load_documents2(wrapper_paths)
    for doc in wrappers:
        doc.metadata['document_type'] = 'wrapper script'
    print(f'There are a total of {len(wrappers)} wrappers. ')


    ## documentation_pdfs and htmls
    # Collect all file paths
    doc_paths = []
    for root, _, files in os.walk(documentations):
        for file in files:
            module_name, ext = os.path.splitext(file)
            if module_name in modules_that_exist:
                full_path = os.path.join(root, file)
                doc_paths.append(full_path)

    # Print or use the full file paths
    documentation_files = load_documents2(doc_paths)
    for doc in documentation_files:
        doc.metadata['document_type'] = 'documentation'
    print(f'There are a total of {len(documentation_files)} documentation_files. ')


    ############## GP website docs
    ## load in genepattern website details:
    urls = [
        'https://genepattern.org/user-guide#gsc.tab=0',
        'https://genepattern.org/quick-start#gsc.tab=0',
        'https://genepattern.org/tutorial#gsc.tab=0',
        'https://genepattern.org/file-formats-guide#gsc.tab=0',
        'https://genepattern.org/administrators-guide#gsc.tab=0',
        'https://genepattern.org/programmers-guide#gsc.tab=0',
        'https://genepattern.org/concepts#gsc.tab=0',
    ]
    metadata = ['User guide', 
                'Quick start', 
                'GenePattern tutorial', 'GenePattern File formats', 'GenePattern Administrators guide',
            'Genepattern programmers guide', 
            'GenePattern Concepts']

    docs = [WebBaseLoader(url).load() for url in urls]



    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100
    )
    guides = text_splitter.split_documents(docs_list)

    print(f'There are a total of {len(guides)} genepattern guides. ')
    
    return manifest_docs, readme_docs, thread_docs, wrappers, documentation_files, guides




