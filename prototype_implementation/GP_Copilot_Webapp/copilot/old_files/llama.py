from langchain_community.document_loaders import PyPDFLoader, html
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
# LangChain supports many other chat models. Here, we're using Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
import chromadb
import os
from datetime import datetime
import pandas as pd
import re

from langchain.text_splitter import MarkdownTextSplitter


## progress
from tqdm import tqdm
# from .query_analysis import *
## parsing R functions

def load_documents(fps):
    """
    Loads multiple document and generates a list of documents for langchain 
    """
    docs = []
    for path in fps:
        try:
            fp = os.path.join(path)
            docs = docs + load_document(fp)
            # os.system(f'rm {fp}')
        except Exception as e:
            # os.system(f'rm {fp}')
            continue
            # print(f'Unsupported file type {path}')
    return docs

def rmd_loader(fp):
    """
    parses Rmd (vignette) file ## using markdown text splitter now 
    """
    # Read the vignette file
    with open(fp, 'r') as file:
        vignette_content = file.read()

######################### old splitter
    # Extract R code chunks
    # r_chunks = re.findall(r'```{r.*?}\n(.*?)```', vignette_content, re.DOTALL)
########################### new splitter 

    splitter = MarkdownTextSplitter(
    chunk_size=1000,  # Size of each chunk
    chunk_overlap=50  # Overlap between chunks
    )

    r_chunks = splitter.split_text(vignette_content)
    
    basename = os.path.splitext(os.path.basename(fp))[0]
    split_text = re.sub(r'[^a-zA-Z0-9]+', ' ', basename)
    # Process each chunk as needed
    page_content =  f"This file is called: {split_text}. \n\n fp: {fp} \n\n Vignette: " +  " ".join(r_chunks)
    name, extension = os.path.splitext(fp)
    metadata = {
        'source' : fp,
        'function_name' : basename,
        'file_name' :os.path.basename(fp),
        'module_name' : fp.split('/')[-2],
        'document_type' : 'vignette',
    }
    
    ## return document
    doc = Document(
        page_content = page_content,
        metadata = metadata
    )
    
    return [doc]

    
def R_loader(fp):
    """
    parses R (script) file. Most likely a function file
    """
    with open(fp, 'r') as file:
        content = file.read()
    file.close()
    
    
    basename = os.path.splitext(os.path.basename(fp))[0]
    split_text = re.sub(r'[^a-zA-Z0-9]+', ' ', basename)
    # Process each chunk as needed
    page_content =  f"This is documentation for the function {split_text}. \n\n fp: {fp} \n\n The function contents:  " + (content)
    name, extension = os.path.splitext(fp)
    metadata = {
        'source' : fp,
        'function_name' : basename,
        'file_name' :os.path.basename(fp),
        'module_name' : fp.split('/')[-2],
        'document_type' : extension,
    }
    
    ## return document
    doc = Document(
        page_content = page_content,
        metadata =metadata
    )
    
    return [doc]
    
    

def load_document(fp):
    """
    Loads a document or multiple documents into vector store
    
    list of loaders: 
    https://api.python.langchain.com/en/latest/community_api_reference.html#module-langchain_community.document_loaders 
    """
    ## load only files I want:
    name, extension = os.path.splitext(fp)
    if extension in ['.R', '.txt', '.Rmd', '.md', '.pdf']:
        for file_ext in ['.R', '.txt', '.Rmd', '.md', '.pdf']:
            if file_ext in fp:
                if ".pdf" == extension:
                    print('hi')
                    raw_documents = PyPDFLoader(fp).load()
                elif ".html" == extension:
                    raw_documents = html.UnstructuredHTMLLoader(fp).load()
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
        doc.metadata['document_type'] = extension

    return docs


def create_db(docs, embedding_function, model = 'llama3'):
    """
    From list of Document objects, create a Chroma vector store using
    llama embeddings as default. Return db and retriever. 
    https://python.langchain.com/v0.1/docs/modules/data_connection/vectorstores/
    """
    now = datetime.now()
    ## Chroma 
    # db = Chroma.from_documents(documents, embedding_function)
    # Create retriever
    
    print(f'\n Vector DB took {datetime.now() - now}')
    
    db = Chroma.from_documents([docs[0]], embedding_function)
    with tqdm(total=len(docs), desc="Ingesting documents") as pbar:
        for d in docs:
            if db:
                db.add_documents([d])
            pbar.update(1)
            
    retriever = db.as_retriever() 
    
    return db, retriever



def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


def create_wrapper_script(model, question, db):
    """
    Runs script to generate wrapper script
    """
    
    prompt = ChatPromptTemplate.from_template("""
        You are a module developer at GenePattern with a PHD in Computer Science and Bioinformatics. \
        You are an expert on common bioinformatics workflows. \
        A GenePattern module is code wrapped around a bioinformatics method or analysis and is made available to be used in the science community. \
        A module contains a wrapper script that parses user arguments from the command line, \
        performs steps to process that data, and then calls the bioinformatics method to perform the analysis. \
        Put the code in three quote brackets.
        Answer the question based on the documentation in the context: {context}
        Question: {question}

    """)
    d
    chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
        )
    analysis = query_analysis(model, question)
    docs = retrieval(db, analysis)
    output = chain.invoke({'context' : format_docs(docs), 'question': question})
    
    
    with open('responses.txt', 'a') as file:
        file.write('*************************************************************************************')
        file.write(f"\n\n {datetime.now()}")
        file.write(output.replace('`', ''))
        file.write('*************************************************************************************')
    file.close()
    
    with open('questions.txt', 'a') as file:
        file.write('*************************************************************************************')
        file.write(f"\n\n {datetime.now()}")
        file.write(question)
        file.write('*************************************************************************************')
    file.close()
    
    
    return output


def create_dockerfile(model, question, wrapper_script):
    """
    Creates a dockerfile from system prompt
    """
    prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant designed to output a Dockerfile. The Dockerfile should be based off a bioconductor/bioconductor_docker image with a specific 
        version. Do not use the latest or devel tags. Pay very close attention to making sure to install the correct version of all dependencies.
        
        In the Dockerfile you should:

        * Make sure to include a FROM command at the beginning of the Dockerfile.
        * Install the specified package from Bioconductor, as well as getopt and optparse. 
        * ADD the file {wrapper_script} to the container in the user home directory.
        * Do not specify either a USER or CMD command at the end of the Dockerfile. 
        * Include only a single RUN commands which performs all installations.
        * Return only the text of the Dockerfile itself. Do not wrap your response in a markdown code tag.
        answer this question: {question}
        """)
    
    chain = prompt | model | StrOutputParser()
    dockerfile = chain.invoke({'wrapper_script':wrapper_script, 'question':question})
    
    with open("Dockerfile", 'a') as file:
        file.write(dockerfile)
    file.close()
    
    return dockerfile



def create_manifest(model, wrapper_script, state):
    MANIFEST_PROMPT = ChatPromptTemplate.from_template("""
    Act as a senior software developer. I am going to describe a GenePattern manifest file, and then I would like you to create one based on information I provide. The first line is a "#" followed by the name of the function I provide. The second line is a # followed by the current date and time. Do not include any blank lines in the output.
    The next section should be pasted verbatim:
---------------------------------------------------------
    JVMLevel=
    LSID=
    author=GenePattern Team + ChatGPT
    commandLine=
    cpuType=any
    The next line should be "description=" followed by a brief description of the function I provided.
    The next line should be "documentationUrl=" followed by the URL to the site where the function is described.
    The next line should be "fileFormat=" followed by a comma-separated list of the file extensions output by the function.
    The next line should be "job.cpuCount="
    The next line should be "job.docker.image=[DOCKER IMAGE HERE]"
    The next section should be pasted verbatim:
    job.memory=
    job.walltime=
    language=any
    The next line should start with "categories=" and should be whichever one of the terms in the following comma-separated list best describes the requested function: alternative splicing,batch correction,clustering,cnv analysis,data format conversion,differential expression,dimension reduction,ecdna,flow cytometry,gene list selection,gsea,image creators,metabolomics,methylation,missing value imputation,mutational significance analysis,pathway analysis,pipeline,prediction,preprocess & utilities,projection,proteomics,rna velocity,rna-seq,rnai,sage,sequence analysis,single-cell,snp analysis,statistical methods,survival analysis,variant annotation,viewer,visualizer
    The next line should be "name=" and then the name of the function.
    The next line should be "os=any"
    After that, identify all the parameters in the wrapper script. 
---------------------------------------------------------
    The manifest file should include each parameter of the provided function, in the format. I will provide below. Here are some instructions for this format:
    1. When you see a # character, replace it with the number of the parameter in the provided function.
    2. When you see "default_value=", place the parameter's default value after the "=" if there is one.
    3. When you see "description=", add the parameter's description after the "="
    4. When you see "name=", add the parameter's name after the "="
    5. When you see "optional=", write "on" if the parameter is optional
    6. In the parameter name, replace the "#" with the cardinal number of the parameter
    7. When you see "flag=", add the parameter's command-line flag after the "=" if there is one. If there are more than one way of specifying a flag, use the one that starts with two hyphens: "--"
    8. When you see "type=", add the parameter's type after the "=". The type should be the term in the following comma-separated list that corresponds most closely to the type of parameter: CHOICE,FILE,Floating Point,Integer,TEXT,java.lang.String. Pick java.io.File if you think the input are filepaths. 
    9. When you see "taskType=", add the type of analysis this module performs. For example: batch correction, visualizer, scRNA analysis. You can infer the category based on the name and description.
    10 If you think a parameter is a file path, put IN for the p#_MODE
    
---------------------------------------------------------
    Here is the format for each parameter:
    p#_MODE=
    p#_TYPE=
    p#_default_value=
    p#_description=
    p#_fileFormat=
    p#_flag=
    p#_name=
    p#_numValues=
    p#_optional=
    p#_prefix=
    p#_prefix_when_specified=
    p#_type=
    p#_value=
    taskType= 
---------------------------------------------------------
    For the commandline, generate a Rscript commandline to run the wrapper script. The parameters should be: --flag <value>. 
    
    Here are some more context: 
    LSID = {lsid}
    name = {module_name}
    src.repo = {source_repo}
    authors = {authors}
    
    
    Use that informaion to create a manifest file based on this wrapper file: {wrapper}
    
-------------------------------------------------------------

    Make sure that the commandline parameter names in the <> brackets match the parameter names in the p#_name= ! 
    """)
    
    chain = MANIFEST_PROMPT | model | StrOutputParser()
    
    manifest = chain.invoke({'wrapper': wrapper_script, 
                             'lsid':state['lsid'], 
                             'module_name':state['module_name'],
                             'source_repo' : state['github_repo'],
                             'authors' : state['authors']})
    
    with open('manifest', 'a') as file:
        file.write(manifest)
    file.close()
    
    return manifest
    
    

def create_paramgroups(model, manifest):
    """
    Use the manifest to create a parameter groups.json file. 
    """
    SYSTEM = ChatPromptTemplate.from_template("""

    You are a GenePattern Module Developer. \n\n
    
    Here's an example manifest file: 
    
    \n\n 
    
    
    #module_dev_pipeline
    #Fri Jun 28 21:13:38 UTC 2024
    JVMLevel=
    LSID=urn\:lsid\:8080.gpserver.ip-172-31-26-71.ip-172-31-26-71.ec2.internal\:genepatternmodules\:752\:12
    author=GP Team, Ollama, OpenAI
    commandLine=python3 /home/main.py --pkg <pkg> --model <model> --analysis "<analysis>" --github_repo <github_repo_url>   <openai.api.key> <assistant.id> <vector.store.id>
    cpuType=any
    description=ADMINS ONLY <br>\nmakes wrapper script, dockerfile, and manifest for a bioconductor package and some analysis. 
    documentationUrl=
    fileFormat=
    job.cpuCount=
    job.docker.image=edwin5588/ai\:v1
    job.memory=
    job.walltime=
    language=any
    name=module_dev_pipeline
    os=any
    p1_MODE=
    p1_TYPE=TEXT
    p1_default_value=
    p1_description=package name
    p1_fileFormat=
    p1_flag=--pkg
    p1_name=pkg
    p1_numValues=0..1
    p1_optional=
    p1_prefix=
    p1_prefix_when_specified=
    p1_type=java.lang.String
    p1_value=
    p2_MODE=
    p2_TYPE=TEXT
    p2_default_value=llama3
    p2_description=model type
    p2_fileFormat=
    p2_flag=--model
    p2_name=model
    p2_numValues=0..1
    p2_optional=
    p2_prefix=
    p2_prefix_when_specified=
    p2_type=java.lang.String
    p2_value=llama3\=llama3;gpt-4o\=gpt-4o
    p3_MODE=
    p3_TYPE=TEXT
    p3_default_value=
    p3_description=
    p3_fileFormat=
    p3_flag=--analysis
    p3_name=analysis
    p3_numValues=0..1
    p3_optional=
    p3_prefix=
    p3_prefix_when_specified=
    p3_type=java.lang.String
    p3_value=
    p4_MODE=
    p4_TYPE=TEXT
    p4_default_value=
    p4_description=link to zip folder for github repo. Steps\: 1. Click on "Code (green button next to add file)", 2. Right click "Download ZIP", 3. Copy link, 4. Paste link here.  
    p4_fileFormat=
    p4_flag=--github_repo
    p4_name=github_repo_url
    p4_numValues=0..1
    p4_optional=on
    p4_prefix=
    p4_prefix_when_specified=
    p4_type=java.lang.String
    p4_value=
    p5_MODE=
    p5_TYPE=TEXT
    p5_default_value=
    p5_description=
    p5_fileFormat=
    p5_flag=--openai_api_key
    p5_name=openai.api.key
    p5_numValues=0..1
    p5_optional=on
    p5_prefix=--openai_api_key
    p5_prefix_when_specified=--openai_api_key
    p5_type=java.lang.String
    p5_value=
    p6_MODE=
    p6_TYPE=TEXT
    p6_default_value=
    p6_description=
    p6_fileFormat=
    p6_flag=assistant_id
    p6_name=assistant.id
    p6_numValues=0..1
    p6_optional=on
    p6_prefix=assistant_id
    p6_prefix_when_specified=assistant_id
    p6_type=java.lang.String
    p6_value=
    p7_MODE=
    p7_TYPE=TEXT
    p7_default_value=
    p7_description=
    p7_fileFormat=
    p7_flag=vector_store_id
    p7_name=vector.store.id
    p7_numValues=0..1
    p7_optional=on
    p7_prefix=vector_store_id
    p7_prefix_when_specified=vector_store_id
    p7_type=java.lang.String
    p7_value=
    privacy=public
    quality=development
    src.repo=
    taskDoc=
    taskType=
    userid=edwin5588
    version=meow?
    
    
    \n\n 
    
    And here's an example paramgroups.json file corresponding to the manifest file: \n\n
    
    [
    {{
        "name": "Required",
        "description": "Group one description.",
        "hidden": false,
        "parameters": [
            "pkg",
            "model",
            "analysis"
        ]
    }},
    {{
        "name": "Optionals",
        "description": "Group two description.",
        "hidden": false,
        "parameters": [
            "github_repo_url",
            "openai.api.key",
            "assistant.id",
            "vector.store.id"
        ]
    }}
]

    Your task is to create a paramgroups.json file for an user provided manifest file.
    Please infer from the parameter names and descriptions to which group to put them into. 
    
    Manifest file: {manifest}
    
    """)
    
    chain = SYSTEM | model | StrOutputParser()
    
    paramgroups = chain.invoke({'manifest': manifest})
    
    return paramgroups