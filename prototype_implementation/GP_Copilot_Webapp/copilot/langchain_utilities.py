from .imports import *

import time 
from PIL import Image
from io import BytesIO
import base64
import os
from langchain_community.embeddings import FastEmbedEmbeddings
from .old_files.llama import *
from uuid import uuid4
from langchain_google_genai import ChatGoogleGenerativeAI
import boto3
from botocore.config import Config
from langchain_aws import BedrockLLM, BedrockEmbeddings
from langchain_ollama import OllamaEmbeddings

def website_to_document(url):
    """
    Downloads a website, returns a bunch of documents to load into vector store retriever
    """

    # Send a GET request to the URL
    response = requests.get(url)
    if response.status_code == 200:
        html_content = response.text
    else:
        print('Failed to retrieve the document')
    soup = BeautifulSoup(html_content, 'html.parser')
    title = soup.title.string
    all_text = soup.get_text()
    links = [a['href'] for a in soup.find_all('a', href=True)]

    data = {
    'title': title,
    'text': all_text,
    'links': links
    }
    document = Document(
    page_content=data['text'],
    metadata={
        'title': data['title'],
        'links': data['links'],
    }
    )
    return document, soup.prettify()

def file_to_documents(fp):
    """
    parses the cloud manage modules page and get a list of modules. 
    """
    # Open and read the HTML file
    with open(fp, 'r', encoding='utf-8') as file:
        html_content = file.read()
    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    # Find all tables
    tables = soup.find_all('table')
    # Find all rows with class "task-title"
    task_rows = soup.find_all("tr", class_="task-title")
    # List to store documents for the vector store
    documents = []
    # Extract task names, descriptions, and links
    for row in task_rows:
        # Get the task name (inside the third <td>)
        task_name = row.find_all("td")[2].contents[0].strip()
        # Get the description (inside <span class="smalltype5">)
        description_span = row.find("span", class_="smalltype5")
        task_description = description_span.get_text(strip=True) if description_span else "No description"
        # Find the next sibling <tr> and extract the link if present
        next_row = row.find_next_sibling("tr")
        link = next_row.find("a", href=True)['href'] if next_row and next_row.find("a", href=True) else "No link"
        # Create a document with the specified namespace
        row = f'''
            Module name is {task_name},
            description is {task_description}, 
            link to module is: https://cloud.genepattern.org{link}
        '''
        document = Document(
            page_content = row,
            metadata={
            "task_name": task_name,
            "description": task_description,
            "category": "module"   
            })
        documents.append(document)
        
    return documents

def get_embeddings(emb_type = 'titan'):
    """
    gets the embeddings based on embedding type
    """
    
    if emb_type == "titan":
        embeddings = BedrockEmbeddings(model_id ='amazon.titan-embed-text-v2:0')
        
    elif emb_type == 'fast':
        embeddings = FastEmbedEmbeddings()
    elif emb_type == 'nomic':
        embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    return embeddings
    
    
    
def get_vector_store(path = '/chroma', session_id = None):
    """
    Gets the retriever from the path (actually, this gets the vector store lol )
    and session id is actually the collection name haha
    """
    embeddings = get_embeddings('fast')
    # Initialize an empty ChromaDB client
    client = chromadb.PersistentClient(path=path)

    # Create or get a collection (you can specify your own name)
    if session_id:
        collection_name = session_id
    else:
        collection_name = "my_vector_store"  # Change this to your desired collection name

    collection = client.get_or_create_collection(name=collection_name)

    # Initialize an empty Chroma vector store with the collection
    vector_store = Chroma(client=client,
                        embedding_function = embeddings,
                        persist_directory = path,
                        collection_name=collection_name)
    
    print(f'Vector store: {collection_name} found. Number of documents in collection: {collection.count()}')
    
    return vector_store

def message_printer(messages):
    """
    Pretty prints the messages
    """
    
    for message in messages:
        print(message.content[-20:])
        

def retrieve_documents(retriever_path, query, collection_names=None, metadata_filter = None):
    """
    Goes through available collections and looks for documents, 
    
    will do retrieval through a multiqueryretriever.
    can take multiple metadata filters
    
    collection_names list[str]--> which collections to look into
    all possible: 
    [
        'genepattern_guide',
        'genepattern_module_manifests',
        'genepattern_module_readmes',
        'genepattern_module_wrappers',
        'genepattern_module_documentations',
        'genepattern_threads'
    ]

    """
    collections = ['genepattern_guide','genepattern_module_manifests','genepattern_module_readmes','genepattern_module_wrappers','genepattern_module_documentations','genepattern_threads']
    docs_to_return = []
    
    if collection_names:
        collections_to_find = collection_names
    else:
        collections_to_find = collections
    
    ## This retrieve prompt isn't used in the retriever_from_llm yet, but we can include it in "MultiQueryRetriever.from_llm" on line 192. 
    retrieve_prompt = ChatPromptTemplate.from_template("""
            You are an AI language model assistant. 
            Your task is to read a vector database query and generate 3 different versions of the query. 
            The query will be related to bioinformatics workflows or platform usage. 
            Provide the alternative questions and queries separated by new lines. 
            
            - If a module is mentioned in a format like `workflow_name.function_name`,
            write an extra query to obtain **all modules from the same workflow** (i.e., all modules starting with `workflow_name.`).
            Original question: {question}
            """)
    
    for collection in collections_to_find:
        vector_store = get_vector_store(retriever_path, collection)
        # construct search kwargs:
        search_kwargs = {
            'k' : 4,
        }
        ## Commented out as the metadata filter takes more time to execute
        ## https://python.langchain.com/docs/concepts/vectorstores/#metadata-filtering 
        # if metadata_filter:
        #     if len(metadata_filter) > 1:
        #         search_kwargs['filter'] = {'$and' : metadata_filter}
        #     else:
        #         search_kwargs['filter'] = metadata_filter[0]

        retriever = vector_store.as_retriever(search_kwargs = search_kwargs)
        retriever_from_llm = MultiQueryRetriever.from_llm(retriever = retriever, 
                                                        llm = get_model('llama-mini'))
        docs = retriever_from_llm.invoke(query)
        docs_to_return += docs
        
    return docs_to_return


def load_files_into_vector_store(vector_store, fps):
    """
    Loads files into the vector , returns retriever
    """
    documents = load_documents(fps)
    
    uuids = [str(uuid4()) for _ in range(len(documents))]
    # Load the documents into the vector store
    vector_store.add_documents(documents, ids = uuids)
    print(f'Loaded {len(documents)} documents into the vector store')

    return vector_store.as_retriever()
    
    
def get_chain(model_type = 'llama3.2', aws=False, retriever_path = "/chroma", custom_system_prompt = '', session_id = None, verbose = False):
    '''
    
    OLD LANGCHAIN THAT IS A RAG CHAIN. 
    Craft the langchain chain for the website to invoke
    '''
    # print(os.path.exists(retriever_path))
    vector_store = get_vector_store(retriever_path, session_id)
    model = get_model(model_type, aws)
    retriever = vector_store.as_retriever()

    if not custom_system_prompt:
        template = """
            You are a bioinformatics expert who works for the GenePattern team.
            Your job is to answer bioinformatics related questions about running a workflow. 
            If an image description is provided, describe the image. 
            
            Do not describe tools that are not in the vector store, instead respond
            with "That tool is not currently available in GenePattern. Feel free to contact
            the GenePattern team if you think it would be a good addition to our repository. Email: edh021@cloud.ucsd.edu"
            Provide input file formats when giving instructions on how to run modules
            or tools. Only give module suggestions for modules in GenePattern.
            Do not tell users to “go to GenePattern and log in”.
            Answer the following questions using all your knowledge
            and providing as much detail as possible with step-by-step instructions.

            If answering questions about workflows, only provide modules that exist on the GenePattern server. 
            """
    else:
        template = custom_system_prompt
        
    template = template + '''
            There may be previous chat messages. Use that to your advantage. 
            
            Use the following context to answer the question: {context}
            
            {image_description}.
            
            The question is: {question}.
            '''
    prompt = ChatPromptTemplate.from_template(template)

    retriever = vector_store.as_retriever()
    configurable_retriever = retriever.configurable_fields(
        search_kwargs=ConfigurableField(
            id="search_kwargs",
            name="Search Kwargs",
            description="The search kwargs to use",
        ) ## remember when loading docs we need to add in a namespace or soemthing
    )

    retriever_from_llm = MultiQueryRetriever.from_llm(retriever = retriever, llm = model)
    if verbose:
        chain = (
            {"context": retriever_from_llm, "question": RunnablePassthrough(), 'image_description' : RunnablePassthrough()}
            | prompt
            | model
        )
    else:
        chain = (
        {"context": retriever_from_llm, "question": RunnablePassthrough(), 'image_description' : RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    return chain

    
    
    
def get_model(model_type, aws = True, image = False):
    """
    Get model based on model_type
    """
    ## model pa
    accesskey = os.environ['aws_access_key']
    secretkey = os.environ['aws_secret_key']
    config = Config(read_timeout = 30000)
    region = 'us-east-1'
    temperature = 0.1
    
    if "llama-3.3" in model_type: 
        if aws:
            # accesskey=os.environ['aws_access_key']
            # secretkey = os.environ['aws_secret_key']
            if image:
                model = ChatBedrockConverse(
                        model_id='us.meta.llama3-2-90b-instruct-v1:0',
                        temperature=temperature,
                        aws_access_key_id=accesskey,
                        aws_secret_access_key=secretkey,
                        region_name = region,
                        config = config
                    )
                # print('Using AWS bedrock llama!, model: us.meta.llama3-2-11b-instruct-v1:0')
            else:
                model = ChatBedrockConverse(
                        model_id='us.meta.llama3-3-70b-instruct-v1:0',
                        temperature=temperature,
                        aws_access_key_id=accesskey,
                        aws_secret_access_key=secretkey,
                        disable_streaming = True,
                        region_name = region,
                        config = config,
                    )
                # print('Using AWS bedrock llama!, model: us.meta.llama3-3-70b-instruct-v1:0')
            
        else:  
            model = ChatOllama(model = 'llama3.2-vision',
                           temperature=temperature)
            
    elif 'llama-mini' in model_type:
        model = ChatBedrockConverse(
                    model='us.meta.llama3-2-1b-instruct-v1:0',
                    temperature=temperature,
                    aws_access_key_id=accesskey,
                    aws_secret_access_key=secretkey,
                    disable_streaming = False,
                    region_name = region,
                    config = config,
                )
        print('Using llama-mini')

    elif 'gemini' in model_type:
        model = ChatGoogleGenerativeAI(
            model = 'gemini-1.5-pro', 
            temperature=temperature, 
            api_key = os.environ['GOOGLE_GEMINI_API_KEY']
            )
        
    elif 'deepseek' in model_type:
        model = ChatBedrockConverse(
            model = 'us.deepseek-llm-r1-distill-llama-70b', 
            temperature = temperature, 
            aws_access_key_id=accesskey,
            aws_secret_access_key=secretkey,
            region_name = region,
            config = config
            )

    elif 'haiku' in model_type:
        model= ChatBedrockConverse(
            model = 'us.anthropic.claude-3-5-haiku-20241022-v1:0', 
            temperature = temperature,
            aws_access_key_id=accesskey,
            aws_secret_access_key=secretkey,
            region_name=region,
            config = config
            )

    elif 'mistral' in model_type:
        model= ChatBedrockConverse(
            model = 'mistral.mistral-large-2402-v1:0', 
            temperature = temperature,
            aws_access_key_id=accesskey,
            aws_secret_access_key=secretkey,
            region_name=region,
            config = config
            )
    elif 'gpt' in model_type:
        ## get api key from environment
        openai_key = os.environ['OPENAI_API_KEY']
        model = ChatOpenAI(model = 'gpt-4o', api_key = openai_key,
                           temperature = temperature)
    return model


def build_image_message(user_input, user_img_path, model_type = 'llama3.2-vision', aws = False):
    """
    Builds a langchain message with text and image
    user_img must be binary
    """
    start = time.time()
    model = get_model(model_type, aws, image = True)
    # Define the maximum allowed dimensions
    max_width = 1024  # Replace with AWS Bedrock's maximum width
    max_height = 1024  # Replace with AWS Bedrock's maximum height

    # Fetch the image
    image = Image.open(user_img_path)
    # Convert the image to RGB if it has an alpha channel (RGBA)
    if image.mode == "RGBA":
        image = image.convert("RGB")
    # Resize the image if it exceeds the maximum dimensions
    if image.width > max_width or image.height > max_height:
        image.thumbnail((max_width, max_height))
    
    # Convert the image to JPEG format
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    image_data = base64.b64encode(buffer.read()).decode("utf-8")
    image_prompt = """
        You might be looking at a very complex image. 
        If dealing with graphs or plots, describe the data and trends with as many details as possible. Colors, symbols, shapes are all very important.
        
        If it is an Amplicon Suite sample plot, apply this: 
        
        The SV view file is a PNG/PDF file displaying the underlying sequence signatures of the amplicon. This image consists of:

        The set of amplicon intervals (x-axis)
        Window-based depth of coverage across the intervals represented as histogram (grey vertical bars)
        Segmentation of the intervals based on coverage and copy number estimate of these segments represented by (horizontal black lines spanning the segment where the y-position of the line represents the copy number)
        Discordant read pair clusters represented as arcs where color represents orientation of the reads. Red: Length discordant in expected orientation (forward-reverse), Brown: Everted read pairs (reverse-forward), Teal: Both reads map to forward strand and Magenta: Both reads map to the reverse strand. Vertical colored lines going to the top of the plot indicate connections to source vertex. Thickness of the arc qualitatively depicts the amount of paired-end read support.
        Bottom panel may represent various annotations on the amplicon intervals where the default view displays oncogene annotations.
        The SV view file may be uploaded to web interface for Cycle view to visualize the cycles in conjunction with the SV view.
        
        - Red edges are 'deletion like' (connect two segments of the genome without changing orientation)
        - Brown edges are 'duplication like' (connect the tail of a segment to the head of another, such as is seen in closure of an ecDNA)
        - Pink and teal edges are inverting SVs. Pink edges traverse in the forward direction before the SV, then in the reverse direction after. Teal edges traverse in the reverse direction before the SV, and the forward direction after.

        additional instructions:
            - make sure to get ALL of the chromosomes
        
        and give an detailed biological interpretation and analysis along with the detailed description.
    """
    if 'llama' in model_type:
        message = HumanMessage(
            content=[
                {"type": "text", "text": image_prompt},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data},
                },
            ],
        )
    else:
        print('decoding image here...')
        message = HumanMessage(
            content=[
                {"type": "text", "text": image_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ],
        )
        
    try:
        resp = model.invoke([message]).content
    except:
        resp = "No image provided"
        
    print(resp)
    end = time.time()
    print(f'Image desc took: {end - start} seconds')
    return resp
