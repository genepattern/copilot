from .imports import *
import pandas as pd
import os
import json
import requests
import numpy as np
from pydantic import BaseModel, Field
from typing import Optional



## langgraph stuff
from .langchain_utilities import *
from .old_files.LLM_langgraph import *
from langgraph.graph import END, StateGraph, START
from typing import Annotated, Literal, Sequence
from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

class GPCopilotState(MessagesState):
    # The add_messages function defines how an update should be processed

    # Default is to replace. add_messages says "append"
    # Some states that we should have: 
    # session_id
    # uploaded_file names and paths
    # uploaded images if any
    # model type
    # retriever_path
    # custom_system_prompt if any
    # verbose for debugging
    session_id : int
    extra_field : int
    model_type : str
    retriever_path : str
    custome_system_prompt : str
    verbose : bool
    retriever_path : str
    action : str
    query_type:str
    query_info : dict
    user_query : str

memory = MemorySaver()

try:
    if os.environ['DJANGO_ENV'] == 'production':
        RETRIEVER_PATH = '/chroma' 
    elif os.environ['DJANGO_ENV'] == 'testing':
        RETRIEVER_PATH = '/chroma'
    else:
        RETRIEVER_PATH = 'chroma'
except:
    RETRIEVER_PATH = 'chroma'

class UserQueryType(BaseModel):
    """Always use this tool to structure your response to the user."""
    user_query_classif: Literal[
        "determine_tools",
        "analysis_guidance",
        "follow_on_suggestions",
        "workflow_outlines",
        "interpret_results",
        "fix_errors",
        "other"
    ] = Field(
        description="The category of the user's question. Must be one of: determine_tools, analysis_guidance, follow_on_suggestions, workflow_outlines, interpret_results, fix_errors, or other."
    )

def agent(state):
    """
    determines what kind of question it is. It's either
    1. bioinformatics / module related 
    or
    2. module creation / workflow related question
    """
    print("---- Determining usage ----")
    state['action'] += "---- Determining usage ----\n"
    template = ChatPromptTemplate.from_template("""
        You are a specialized AI assistant for GenePattern, a bioinformatics platform, called GP (GenePattern) Copilot.
        Your job is to categorize user questions into one of the following categories based on the user's intent:
    
        1. **determine_tools** – The user is trying to identify which GenePattern modules or tools are appropriate for their analysis.
            - Examples: 
                - "What tools can I use to analyze RNA-seq data?"
                - "Which tools can I use to visualize gene expression data?"
                - "What tools can I use to perform differential gene expression analysis?"
        2. **analysis_guidance** – The user is seeking detailed advice on how to correctly perform a particular analysis or set parameters.
            - Examples:
                - "How do I set the parameters for the GSEA module?"
                - "How do I interpret the results of the Hierarchical Clustering module?"
        3. **follow_on_suggestions** – The user wants suggestions for what additional analyses they can do after completing an initial step.
            - Examples:
                - "I have run the GSEA module. What other analyses can I do with the results?"
                - "I have performed differential gene expression analysis. What should I do next?"
        4. **workflow_outlines** – The user wants to understand the outline or steps of an entire analysis workflow.
            - Examples:
                - "Can you provide a workflow for analyzing RNA-seq data?"
                - "What are the steps involved in performing differential gene expression analysis?"
                - "Teach me how to run something on GenePattern".
        5. **interpret_results** – The user needs help in interpreting or making sense of analysis results.
            - Examples:
                - "How do I interpret the results of the GSEA module?"
                - "What do the results of the Hierarchical Clustering module mean?"
                - "Here's an image from the HeatMapViewer module. What does it show?"
        6. **fix_errors** – The user is reporting errors or asking how to fix a problem in GenePattern.
            - Examples:
                - "I'm getting an error when running the GSEA module. How can I fix it?"
                - "The Hierarchical Clustering module is not working as expected. What should I do?"
        7. **biological_discussion** – Any other question related to their use of GenePattern that doesn’t fall into the above categories.
            - Examples:
                - "How do I save my analysis results?"
                - "Can you help me understand the GenePattern interface?"
                - "What is a GCT file?"
        8. **other** 
        Here is the user question:
    
        {question}
    
        Additional instructions:
        Only return one of the following values: 
        "determine_tools", "analysis_guidance", "follow_on_suggestions", "workflow_outlines", "interpret_results", "fix_errors", or "other".
        """)
    
    messages = state['messages']
    llm = get_model('haiku', aws = True)
    chain = template | llm.with_structured_output(UserQueryType)
    response =  chain.invoke({'question':messages})
    print(response)
    state['action'] += f"{response}\n"
    ## updates the query type
    state['user_query'] = messages[-1].content
    state['query_type'] = response.user_query_classif
    print(f'----Query type: {response.user_query_classif}----')
    state['action'] += f"----Query type: {response.user_query_classif}----\n"
    state['action'] += "went through Agent \n"
    
    print(state['messages'])
    return {'query_type' : response.user_query_classif, 'user_query' : messages[-1].content, 'action' : state['action'] + f"agent categorized: {response.user_query_classif}"}


def generate_response(state):
    """
    Generates the answer, will identify the modules that relate to solving the user question
    """
    template = """
    You are a bioinformatics expert from the GenePattern team. 
    Your job is to identify GenePattern modules that can accomplish the user's query or workflow needs.

    Sometimes, the user may be asking a follow-up question. 
    Always consider any prior context and messages when formulating your answer.

    If the user is asking about how to run a *user-specified analysis*, you can run the following modules on GenePattern:
    Do NOT suggest modules that are not found in the vector store.

    Be detailed and helpful, especially for users without a strong programming background.

    Only answer with: "I think these [list of modules] modules will help."

    ---

    The user may have already asked previous questions. Take those into account when answering.

    The current question is: {question}

    Context retrieved from the system: {context}
"""

    print('---- Generating Response ----')
    state['action'] += "---- Generating Response ----\n"
    print(f'user query is here: {state["user_query"]}')
    vector_db_query = f"""
            Here's the user query: {state['user_query']}
            There may be extra modules that may help with preprocessing steps or QC. 
            Here's some additional information from the query: {json.dumps(state['query_info'])}
    """
    retrieved_docs = retrieve_documents(RETRIEVER_PATH, 
                                        vector_db_query, 
                                        collection_names = ['genepattern_module_manifests',
                                                            'genepattern_module_documentations'])
    print("--- Docs retrieved ---")
    state['action'] += "--- Docs retrieved ---\n"
    print("number of docs retrieved: ", len(retrieved_docs))
    state['action'] += "number of docs retrieved: " + str(len(retrieved_docs)) + "\n"
    model = get_model('llama-3.3')
    doc_texts = [doc.page_content for doc in retrieved_docs]
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = prompt | model | StrOutputParser()
    response = rag_chain.invoke({'question': state['messages'], 'context': doc_texts})
    return {'messages': [response], 'action' : state['action'] + 'went through generate responses'}

class QueryClassification(BaseModel):
    """
    Extracted GenePattern module information from the AI response.
    """
    detected_modules: list[str] = Field(default_factory=list, description="List of all detected GenePattern modules based on exact or workflow-based matches.")
    workflows_detected: Optional[list[str]] = Field(default=None, description="List of detected workflow base names if applicable (e.g., 'rnaseq').")
    
def detect_modules(state):
    """
    From a list of tools to pick from, 
    """
    print("---- Detecting modules (if any) in user prompt. ----")
    state['action'] += "---- Detecting modules (if any) in user prompt. ----\n"
    prompt = ChatPromptTemplate.from_template("""
        You will receive an AI-generated message that may refer to one or more GenePattern modules.
        
        Often, modules may be part of larger workflows. Module names may follow a pattern like:
        *workflow_name*.*function_name* (e.g., `rnaseq.alignment`, `rnaseq.qc`, `rnaseq.counts`).
        
        Your task is:
        1. Extract the names of all GenePattern modules mentioned or implicitly referred to in the AI response.
        - If a module is mentioned in a format like `workflow_name.function_name`,
            collect **all modules from the same workflow** (i.e., all modules starting with `workflow_name.`).
        - Only extract exact matches from the provided module list.
        - If a name doesn’t exist in the list, skip it.

        You will be provided with 2 pieces of information, one is the list of available modules along with how many times they've been ran this past 10 years. 
        
        The other is the AI response that you need to extract the modules from.
        
        List of modules: 
        ['GISTIC_2.0', 'GSEA', 'ssGSEA', 'GSEAPreranked', 'PreprocessDataset', 'HierarchicalClustering', 'ComparativeMarkerSelection', 'SubMap', 'NearestTemplatePrediction', 'DESeq2', 'MutSigCV', 'HeatMapViewer', 'ssGSEAProjection', 'PredictionResultsViewer', 'HeatMapImage', 'ClsFileCreator', 'ExpressionFileCreator', 'HierarchicalClusteringViewer', 'ComparativeMarkerSelectionViewer', 'NMFConsensus', 'FcsToCsv', 'HierarchicalClusteringImage', 'NMF', 'GEOImporter', 'PTM_SEA', 'MergeHTSeqCounts', 'CsvToFcs', 'CollapseDataset', 'txt2odf', 'ConvertLineEndings', 'KNNXValidation', 'ConsensusClustering', 'MethylationCNVAnalysis', 'SelectFeaturesRows', 'WeightedVoting', 'DifferentialExpressionViewer', 'TCGA.SampleSelection', 'FeatureSummaryViewer', 'AffySTExpressionFileCreator', 'Picard.SortSam', 'DAPPLE', 'TopHat', 'KMeansClustering', 'ExtractComparativeMarkerResults', 'WeightedVotingXValidation', 'KNN', 'ComBat', 'CART', 'ESPPredictor', 'CoxRegression', 'TCGAImporter', 'SelectFeaturesColumns', 'DownloadURL', 'Hisat2Aligner', 'SVM', 'MultiplotStudio', 'Hisat2Indexer', 'SOMClustering', 'PreprocessReadCounts', 'VoomNormalize', 'PCA', 'ConstellationMap', 'ssGSEA.ROC', 'OpenCRAVAT', 'DiSCoVER', 'igv.js', 'SNPFileCreator', 'FastQC', 'ABasicModule', 'ClassNeighbors', 'MutPanning', 'SOMClusterViewer', 'FLAMEPreprocess', 'AmpliconSuite', 'testPipelineGolubNoViewers', 'Salmon.Quant', 'JavaTreeView', 'CopyNumberDivideByNormals', 'SnpViewer', 'CARTXValidation', 'GSEALeadingEdgeViewer', 'ssGSEA_ROC', 'XChromosomeCorrect', 'TCGA.ssGSEA.ROC', 'ImputeMissingValues.KNN', 'PCAViewer', 'CellFie', 'ARACNE', 'Seurat.QC', 'HISAT2.aligner', 'Fpkm_trackingToGct', 'PclToGct', 'Salmon.Indexer', 'HTSeq.Count', 'Picard.BamToSam', 'ScanpyUtilities', 'scVelo', 'Seurat.BatchCorrection', 'GctToPcl', 'VennDiagram', 'IlluminaExpressionFileCreator', 'Cuffdiff', 'ExprToGct', 'Amaretto', 'ComBat_Seq', 'STAR.aligner', 'tximport.DESeq2', 'Kallisto', 'Trimmomatic', 'PreviewFCS', 'Cufflinks', 'PreprocessVelocityTranscriptome', 'RNASeQC', 'RenameFile', 'Seurat.IntegrateData', 'GeneNeighbors', 'Seurat.Preprocessing', 'HISAT2.indexer', 'Convert.Alevin', 'MergeColumns', 'CoGAPS', 'SelectFileMatrix', 'RankNormalize', 'TransposeDataset', 'Bowtie.aligner', 'CBS', 'MetageneProjection', 'Conos.Preprocess', 'Picard.FastqToSam', 'BWA.aln', 'MergeFCSDataFiles', 'DiffEx', 'UniquifyLabels', 'SeuratClustering', 'GeneListSignificanceViewer', 'Seurat.Clustering', 'NMFClustering', 'SparseHierarchicalClustering', 'STAR.indexer', 'Read_group_trackingToGct', 'download_from_gdc', 'SNPMultipleSampleAnalysis', 'SurvivalCurve', 'Seurat.VisualizeMarkerExpression', 'IGV', 'AmpliconSuiteAggregator', 'DeIdentifyFCS', 'TestStep', 'ExtractRowNames', 'STREAM.Preprocess', 'Conos.Cluster', 'RemoveMissingValues', 'Bowtie.indexer', 'sleep', 'SurvivalDifference', 'Picard.MarkDuplicates', 'CytoscapeViewer', 'Picard.SamToFastq', 'RandomForest', 'TestFileList', 'ConumeeCNVAnalysis', 'SplitDatasetTrainTest', 'ExtractFCSDataset', 'SetFCSKeywords', 'MinfiPreprocessing', 'MAGETABImportViewer', 'STREAM.DimensionReduction', 'ReorderByClass', 'MAGEMLImportViewer', 'STREAM.Plot2DVisualization', 'Picard.SamToBam', 'devCoGAPS', 'BWA.indexer', 'DietSeurat.QC', 'SAMtools.FastaIndex', 'Picard.CreateSequenceDictionary', 'STREAM.FeatureSelection', 'MINDY', 'BedToGtf', 'PyCoGAPS', 'Kallisto.Quant', 'MergeRows', 'CommunityAmaretto', 'Harmony']

        AI Response:
        {ai_response}

        Additional Notes:
        - Do not perform fuzzy matching or corrections.
        - Do not include modules that are not part of the available list.
        - When detecting a module like `rnaseq.analyze`, you must return all modules starting with `rnaseq.` from the module list.

        To respond, give me a json format of: 
        "workflow name 1" : [module 1, module 2, ...], 
        "workflow name 2" : [module 1, module 2, ... ], 
        ...
        "workflow name n" : [module 1, module 2, ... ], 

        just the json format is fine, no additional brackets or anything other than the dictionary brackets. 

        - Make sure the module you identified exists in the list of available modules! 
    """)
    llm = get_model('llama-3.3', aws = True)
    chain = prompt | llm 
    # print(f"---- User prompt: ---- \n {state['messages'][-1].content}----")
    response = chain.invoke({
        'ai_response': state['messages'][-1].content
    })
    # print(response)
    modules_identified = response.content
    state['query_info'] = modules_identified
    print(modules_identified)
    return {'query_info': modules_identified, 'action' : state['action'] + "went through detect modules."}


class validation(BaseModel):
    """
    Validation from the validation bot, structured output
    """

    feedback : Optional[str] = Field(description = "Your feedback to the ai response v.s. the real answer")
    score: Optional[int] = Field(description = "The score you're giving to the ai response, out of 100. ")
    refinement: Optional[str] = Field(description = "Refined AI response.")
    metadata : Optional[dict] = Field(description = 'Any other metadata including i/o tokens and stuff')
    
def validation_bot(state):
    """
    This bot will take some AI response, some user query, and a real response to the question and
    give a score as well as how good an AI response is. 
    """


    system_prompt = """
    You are an expert in assisting biologists with using GenePattern, assume your audience is an undergraduate biology researcher. 

    Your goal is to take some AI identified list of modules and create a workflow for the user to run genepattern. 
    
    Some notes: 
    - Assume the user is starting with raw counts, .csv, .tsv, or some unpreprocessed data. This means there may be additional modules to run before the main analysis. 
    - The workflow should look something like: 
        1. Log into GenePattern (https://cloud.genepattern.org)
        2. Search for module A
        - For module A, provide specific instructions, such as:
            - parameter name 1: file format or input value, ... 
            - parameter name 2: format, value, ... 
        3. Then proceed to module B...
        - Repeat instructions as above
    - Contains the correct parameter names for each module. 
    - Clearly structured.
    - Context-aware (in case the query is a follow-up).

    Here's a list of steps for a general bioinformatics analysis workflow, to help you when you construct your workflow: 
        1.	Experimental Design
        2.	Sample Collection & Data Generation
        3.	Raw Data Acquisition
        4.	Quality Control
        5.	Read Alignment / Mapping
        6.	Post-Alignment Processing
        7.	Quantification / Feature Counting
        8.	Normalization & Statistical Analysis
        9.	Functional Analysis
        10.	Visualization
        11.	Data Integration
        12.	Reporting & Result Sharing
        
    Make sure the ordering of analysis steps are correct. 

    You will receive the following information:
    (1) Original user question
    (2) AI-identified modules
    (3) Dictionary of modules identified
    (4) Supporting documents about the modules

    
    Your output should follow this format:
    - Workflow Steps in the correct order:
        - this means starting from a raw file, processing it, and inputting into analysis. 
    - Answer in markdown format for easy readability.
    - Format nicely.
    
    --- INPUT DATA ---
    
    User Question:
    {user_q}
    
    AI Response:
    {ai_response}
    
    Modules Identified:
    {identified_modules}
    
    Supporting Documents:
    {docs}
    """

    prompt = ChatPromptTemplate.from_template(system_prompt)
    
    vector_db_query = f"""
        modules identified: {state['query_info']}
        """
    retrieved_docs = format_docs(retrieve_documents(RETRIEVER_PATH,
                                                    vector_db_query, 
                                                    ['genepattern_module_documentations']))
    
    for doc in retrieved_docs:
        if 'Seurat.QC' in doc:
            print(doc[:60])
    model = get_model(state['model_type'], True)
    chain = {'user_q': RunnablePassthrough(), 
         'ai_response' : RunnablePassthrough(), 
         'identified_modules' : RunnablePassthrough(), 
         'docs':RunnablePassthrough()} | prompt | model
    
    print(state['messages'][-1])

    response = chain.invoke({'user_q' : state['user_query'],
                            'ai_response' : state['messages'][-1], 
                             'identified_modules' : state['query_info'],
                            'docs' : retrieved_docs})
    
    state['messages']
    return {'messages':[response], 'action' : state['action'] + "went through validation bot"}

def module_factory_node(state):
    """
    direct graph state to module factory
    
    """
    print("---- Starting module factory process ----")
    state['action'] += "---- Starting module factory process ----\n"
    print("---- Module creation complete ----")
    state['action'] += "---- Module creation complete ----\n"
    return state


def tool_recommender(state):
    '''
    Recommends tools to use for cases:
    
    1. **determine_tools** – The user is trying to identify which GenePattern modules or tools are appropriate for their analysis.
    2. **analysis_guidance** – The user is seeking detailed advice on how to correctly perform a particular analysis or set parameters.
    '''
    print("---- Tool Recommender Agent Triggered ----")
    state['action'] += "---- Tool Recommender Agent Triggered ----\n"
    system_prompt = """
        You are a tool recommender for GenePattern.
        A tool is also a "module" on GenePattern. 
        
        You'll receive:
        1. User query, which might be a follow up question, or a biological question. 
        2. List of available modules and their run frequencies
        
        Notes:
        - Your main task is to suggest appropriate modules from the available list to help with the user query.
        - You should also answer any questions about modules. 
 
        User query: 
        {user_query}

        Only recommend modules from this list, and use this list to help you choose what modules to suggest. 
        'GISTIC_2.0 was ran 68430 times,GSEA was ran 60689 times,ssGSEA was ran 56689 times,GSEAPreranked was ran 55149 times,PreprocessDataset was ran 23299 times,HierarchicalClustering was ran 21443 times,ComparativeMarkerSelection was ran 19012 times,SubMap was ran 18648 times,NearestTemplatePrediction was ran 18165 times,DESeq2 was ran 16136 times,MutSigCV was ran 14475 times,HeatMapViewer was ran 12541 times,PredictionResultsViewer was ran 11326 times,HeatMapImage was ran 10209 times,ClsFileCreator was ran 10134 times,ExpressionFileCreator was ran 8285 times,HierarchicalClusteringViewer was ran 7636 times,NMFConsensus was ran 7291 times,FcsToCsv was ran 7060 times,HierarchicalClusteringImage was ran 6981 times,NMF was ran 6442 times,GEOImporter was ran 5253 times,PTM_SEA was ran 4625 times,MergeHTSeqCounts was ran 4565 times,CsvToFcs was ran 4360 times,CollapseDataset was ran 4277 times,txt2odf was ran 4147 times,ConvertLineEndings was ran 3972 times,KNNXValidation was ran 3941 times,ConsensusClustering was ran 3808 times,MethylationCNVAnalysis was ran 3609 times,SelectFeaturesRows was ran 3596 times,WeightedVoting was ran 3047 times,DifferentialExpressionViewer was ran 2957 times,TCGA.SampleSelection was ran 2926 times,FeatureSummaryViewer was ran 2914 times,AffySTExpressionFileCreator was ran 2903 times,Picard.SortSam was ran 2858 times,DAPPLE was ran 2836 times,KMeansClustering was ran 2606 times,ExtractComparativeMarkerResults was ran 2422 times,WeightedVotingXValidation was ran 2379 times,KNN was ran 2370 times,ComBat was ran 2302 times,CART was ran 2207 times,ESPPredictor was ran 2132 times,CoxRegression was ran 2064 times,TCGAImporter was ran 1951 times,SelectFeaturesColumns was ran 1921 times,DownloadURL was ran 1873 times,SVM was ran 1829 times,MultiplotStudio was ran 1809 times,SOMClustering was ran 1691 times,VoomNormalize was ran 1543 times,PCA was ran 1543 times,ConstellationMap was ran 1402 times,ssGSEA.ROC was ran 1390 times,OpenCRAVAT was ran 1384 times,DiSCoVER was ran 1362 times,igv.js was ran 1293 times,SNPFileCreator was ran 1104 times,FastQC was ran 1077 times,ClassNeighbors was ran 1023 times,MutPanning was ran 968 times,SOMClusterViewer was ran 864 times,FLAMEPreprocess was ran 856 times,AmpliconSuite was ran 795 times,Salmon.Quant was ran 772 times,JavaTreeView was ran 764 times,CopyNumberDivideByNormals was ran 760 times,SnpViewer was ran 741 times,CARTXValidation was ran 733 times,GSEALeadingEdgeViewer was ran 729 times,XChromosomeCorrect was ran 715 times,TCGA.ssGSEA.ROC was ran 702 times,ImputeMissingValues.KNN was ran 645 times,PCAViewer was ran 630 times,CellFie was ran 603 times,ARACNE was ran 561 times,Seurat.QC was ran 559 times,HISAT2.aligner was ran 535 times,Fpkm_trackingToGct was ran 532 times,PclToGct was ran 495 times,Salmon.Indexer was ran 494 times,HTSeq.Count was ran 488 times,Picard.BamToSam was ran 469 times,ScanpyUtilities was ran 390 times,scVelo was ran 386 times,GctToPcl was ran 351 times,VennDiagram was ran 348 times,IlluminaExpressionFileCreator was ran 340 times,ExprToGct was ran 312 times,Amaretto was ran 308 times,ComBat_Seq was ran 306 times,STAR.aligner was ran 304 times,tximport.DESeq2 was ran 291 times,Trimmomatic was ran 273 times,PreviewFCS was ran 265 times,PreprocessVelocityTranscriptome was ran 246 times,RNASeQC was ran 241 times,RenameFile was ran 221 times,Seurat.IntegrateData was ran 220 times,GeneNeighbors was ran 218 times,Seurat.Preprocessing was ran 213 times,HISAT2.indexer was ran 213 times,Convert.Alevin was ran 202 times,MergeColumns was ran 199 times,CoGAPS was ran 196 times,SelectFileMatrix was ran 189 times,RankNormalize was ran 188 times,TransposeDataset was ran 188 times,MetageneProjection was ran 183 times,Conos.Preprocess was ran 178 times,Picard.FastqToSam was ran 172 times,BWA.aln was ran 171 times,MergeFCSDataFiles was ran 159 times,UniquifyLabels was ran 148 times,GeneListSignificanceViewer was ran 126 times,Seurat.Clustering was ran 116 times,NMFClustering was ran 112 times,SparseHierarchicalClustering was ran 110 times,STAR.indexer was ran 107 times,Read_group_trackingToGct was ran 107 times'

        Supporting docs if available:
        {docs}
        """   
    
    
    vector_db_query = f"""
    Here's the user query: {state['user_query']}
    Here's additional classification: {state['query_type']}
    """
    
    retrieved_docs = retrieve_documents(RETRIEVER_PATH, 
                                        vector_db_query, 
                                        collection_names = ['genepattern_module_manifests',
                                                            'genepattern_module_documentations'])
    doc_texts = [doc.page_content for doc in retrieved_docs]
    prompt = ChatPromptTemplate.from_template(system_prompt)
    rag_chain = prompt | get_model(state['model_type'], aws=True) | StrOutputParser()
    response = rag_chain.invoke({
        "user_query": state['messages'][-4:],
        "docs": doc_texts
    })
    
    return {"messages": [response], 'action' : state['action'] + "---- Tool Recommender Agent Triggered ----\n"}

def analysis_interpretation(state):
    """
    For the classification: 'interpret_results'
    
    This function helps users understand the results of a GenePattern analysis module.
    The input may include an image (e.g., heatmap, clustering plot) or a textual job output/result.
    
    The system should:
    - Interpret and explain the result clearly, assuming the user may not have a bioinformatics background.
    - Include biological context where appropriate.
    - If an image was submitted, try to explain what it likely represents and what typical patterns or features users should look for.
    - If a textual result (like a summary or metrics), explain each part of it.
    """
    
    print("---- Interpretation Agent Triggered ----")
    state['action'] += "---- Interpretation Agent Triggered ----\n"
    system_prompt = """
    You are a result interpretation assistant for GenePattern.
    
    The user might ask a follow up question, so answer accordingly. 
    
    The user has submitted either an output file (text result) or a visual image (e.g., from a heatmap, PCA plot, hierarchical clustering, etc.) and wants help understanding it.
    
    Your job is to:
    1. Interpret the results in biological terms.
    2. Explain each component clearly, especially for users without strong bioinformatics background.
    3. If an image is included, explain what kind of output this likely is, what features it typically shows, and how to interpret it.
    4. If it's a result summary (text), break down each metric or result and explain what it means.
    
    User Query:
    {user_query}
    
    Chat history: 
    {history}
    
    
    Retrieved Supporting Documents (optional):
    {context}
    
    Image (if applicable): Describe or reference the image if mentioned in user_query.
    """
    
    vector_db_query = f"""
        Here's the user query: {state['user_query']}
        Classification: {state['query_type']}
    """
    
    retrieved_docs = retrieve_documents(RETRIEVER_PATH, 
                                        vector_db_query, 
                                        collection_names = ['genepattern_module_manifests',
                                                            'genepattern_module_documentations'])
    doc_texts = [doc.page_content for doc in retrieved_docs]
    
    prompt = ChatPromptTemplate.from_template(system_prompt)
    rag_chain = prompt | get_model(state['model_type'], aws=True) | StrOutputParser()
    
    response = rag_chain.invoke({
        "user_query": state['user_query'],
        'history': state['messages'][-4:],
        "context": doc_texts
    })
    
    return {"messages": [response], 'action' : state['action'] + "went through analysis interpretation \n"}

def other_node(state):
    
    """
    This node will give whatever responses, to the "fix errors" or "other" node. 
    
    simple rag llm that will find some docs based on user query, load it into the prompt and give an output
    """
    prompt = ChatPromptTemplate.from_template("""
        You are a bioinformatics expert to help with questions that biologists might have. 
        When crafting your answer, explain as if you're talking to an undergraduate researcher.
        You will get an entire chat history, along with the user prompt to help with your response.
        
        The user question might be a follow up question. 
        
        Here's the user question: 
        {input}
        
        Chat history is here: 
        {chat_history}
        
        Here are some documentation and context to help: 
        {context}
        
        """)
    
    print(f'---------- Im in the "other" node ----------------------------')
    state['action'] += "---- Other Agent Triggered ----\n"
    messages = state['messages']
    user_query = state['user_query']
    docs = retrieve_documents(RETRIEVER_PATH, user_query, collection_names = ['genepattern_guide', 
                                                                              'genepattern_module_readmes', 
                                                                              'genepattern_threads'])
    formatted = [doc.page_content for doc in docs]
    print(f'Retrieved {len(formatted)} docs!')
    chain = prompt | get_model(state['model_type'])
    response = chain.invoke({'input' : user_query,
                            'chat_history' : messages[-4:],
                            'context' : formatted})
    
    return {'messages' : [response], 'action' : state['action'] + "went through other node \n"}

def fix_errors(state):
    """
    This node will use tools to fix errors by accessing GP job results via api 
    TO BE IMPLEMENTED
    """
    return {'messages': [AIMessage(content = "The debugging jobs feature is coming soon!")]}

def route_from_agent(state):
    """
    Routes to the appropriate node based on the query_type.
    """
    query_type = state.get('query_type')
    if query_type in ['determine_tools', 'follow_on_suggestions']:
        return 'tool_recommender'
    elif query_type == 'workflow_outlines':
        return 'generate_response'
    elif query_type in ['interpret_results', 'analysis_guidance']:
        return 'analysis_interpretation'
    elif query_type in ['fix_errors']:
        return 'fix_errors'
    elif query_type in ['biological_discussion']:
        return 'tool_recommender'
    elif query_type in ['other']:
        return 'other_node'
    else:
        print(f"[WARNING] Unexpected query_type: {query_type}, defaulting to 'generate_response'")
        state['action'] += f"[WARNING] Unexpected query_type: {query_type}, defaulting to 'generate_response'\n"
        return 'tool_recommender'

def get_graph():
    """Builds the LangGraph chatbot pipeline with state initialization and tool handling."""
    graph_builder = StateGraph(GPCopilotState)
    ############ Nodes ############
    graph_builder.add_node("agent", agent)
    graph_builder.add_node('module_factory', module_factory_node)
    graph_builder.add_node("detect_modules", detect_modules)  # Ensure node exists
    
    graph_builder.add_node("generate_response", generate_response)
    graph_builder.add_node("validation_bot", validation_bot)
    graph_builder.add_node('tool_recommender', tool_recommender)
    graph_builder.add_node('other_node', other_node)
    graph_builder.add_node('analysis_interpretation', analysis_interpretation)
    graph_builder.add_node('fix_errors', fix_errors)
    
    ############ Edges ############
    graph_builder.add_edge('module_factory', END)
    graph_builder.add_edge('generate_response', 'detect_modules')
    graph_builder.add_edge('detect_modules', 'validation_bot')
    graph_builder.add_edge('validation_bot', END)
    graph_builder.add_edge('generate_response', END)
    
    graph_builder.add_edge(START, "agent")

    ###########################################
    
    ############ Conditonal Edges ############
    
    graph_builder.add_conditional_edges(
        "agent",
        route_from_agent,
        {
            "module_factory": "module_factory",
            "generate_response": "generate_response",
            "tool_recommender": "tool_recommender",
            "analysis_interpretation": "analysis_interpretation",
            'fix_errors' : 'fix_errors',
            'other_node' : 'other_node'
        }
    )
    #############################################
    return graph_builder.compile(checkpointer=memory)