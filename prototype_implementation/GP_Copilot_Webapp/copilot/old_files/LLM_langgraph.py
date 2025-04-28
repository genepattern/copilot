from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools.retriever import create_retriever_tool
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict
from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START
from ..langchain_utilities import * 
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AnyMessage,ToolMessage 
import json
from langgraph.prebuilt import tools_condition

try:
    if os.environ['DJANGO_ENV'] == 'production':
        RETRIEVER_PATH = '/chroma' 
    else:
        RETRIEVER_PATH = 'chroma'
except:
    RETRIEVER_PATH = 'chroma'


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
    retrieved_docs : list[str]
    user_query: str  # NEW FIELD

memory = MemorySaver()


def generate_response(state, config):
    """
    Generates the answer 
    """
    
    print('---- Generating Response ----')
    template = """
        You are a bioinformatics expert who works for the GenePattern team.
        Your job is to answer bioinformatics related questions about running a workflow. 
        If an image description is provided, describe the image. 
        
        Do not describe tools that are not in the vector store, instead respond
        with "That tool is not currently available in GenePattern. Feel free to contact
        the GenePattern team if you think it would be a good addition to our repository. Email: gp-team@broadinstitute.org"
        Provide input file formats when giving instructions on how to run modules
        or tools. Only give module suggestions for modules in GenePattern.
        Do not tell users to “go to GenePattern and log in”.
        Answer the following questions using all your knowledge
        and providing as much detail as possible with step-by-step instructions.

        If answering questions about workflows, only provide modules that exist on the GenePattern server. 
        """
    template = template + '''
            There may be previous chat messages and context. Use that to your advantage. 
            
            The question is: {question}.
            
            Some context to use: {context}.
            '''
            
    model_type = config.get("configurable", {}).get("model_type", "llama-3.2")
    retrieved_docs = state.get("retrieved_docs", [])  # Read retrieved documents from state
    user_query = state.get("user_query", "")  # Read stored user query
    
    print("--- Docs retrieved ---")
    print(RETRIEVER_PATH)
    
    model = get_model(model_type, aws=True)
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = prompt | model | StrOutputParser()
    
    response = rag_chain.invoke({'question': user_query, 'context': retrieved_docs})  # Use stored query

    return {'messages': [response]}

def get_tools():
    """
    Gets a list of tools to use, for now as of 02/26/2025, we only have the retriever tool.
    """
    
    # retriever_path = config.get("configurable", {}).get("retriever_path", "chroma")
    retriever = get_vector_store(RETRIEVER_PATH).as_retriever()
    retriever.search_kwargs = {"k": 15}
    retriever_from_llm = MultiQueryRetriever.from_llm(retriever = retriever, llm = get_model('llama', True))
    retriever_tool = create_retriever_tool(
        retriever_from_llm,
        "retrieve_gp_docs",
        "Search and return information about GenePattern and related modules."
    )
    return [retriever_tool]


def retriever_node(state, config):
    retriever = get_vector_store(RETRIEVER_PATH).as_retriever()
    retriever.search_kwargs = {"k": 15}
    
    retriever_from_llm = MultiQueryRetriever.from_llm(retriever=retriever, llm=get_model('llama', True))
    
    query = state["messages"][-1].content  # Extract user query
    retrieved_docs = retriever_from_llm.invoke(query)

    # Extract and store document texts
    doc_texts = [doc.page_content for doc in retrieved_docs]
    
    return {
        "messages": state["messages"],
        "retrieved_docs": doc_texts,  # Store retrieved docs
        "user_query": query  # Store user query explicitly
    }


def agent(state, config):
    """
    Determines what kind of question the user is asking and determines what tools to use. 
    """
    print("---- Starting... ----")
    messages = state['messages']
    template = ChatPromptTemplate.from_template("""
        Your job is to determine what to do next. Call tools to help with user query, 
        or generate a response based on the query. 
        
        **User Question:** {question}
    """)

    llm = get_model('llama', aws = True)
    chain = template | llm.bind_tools(get_tools())
    response =  chain.invoke({'question':messages})
    print(response)
    return {'messages': [response]}

def make_decision(state):
    """
    Decision to generate response or retriever
    """
    return 'tools'

        
def get_graph():
    """Builds the LangGraph chatbot pipeline with state initialization and tool handling."""
    graph_builder = StateGraph(GPCopilotState)

    # Add nodes
    graph_builder.add_node("agent", agent)
    graph_builder.add_node("retriever", retriever_node)  # Use retriever_node instead of ToolNode
    graph_builder.add_node("generate_response", generate_response)

    # Define flow
    graph_builder.add_edge(START, "agent")  
    graph_builder.add_conditional_edges("agent", 
                                        make_decision, 
                                        {'tools': 'retriever', 'generate_response': 'generate_response'})
    graph_builder.add_edge('retriever', 'generate_response')
    graph_builder.add_edge("generate_response", END)

    return graph_builder.compile(checkpointer=memory)



def stream_graph_updates(user_input: str):
    config = {
        "configurable": {
            "thread_id": "1",
            "model_type": "haiku",  # Example model type
            "retriever_path": "GP_Copilot/chroma",  # Example retriever
            "custom_system_prompt": "",
            "verbose": True
        }
    }
    
    initial_state = {
        "session_id": "1",
        "messages": [{"role": "user", "content": user_input}]
    }

    graph = get_graph()
    
    for event in graph.stream(initial_state, config):  # Pass both state and config
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

if __name__=="__main__":
    config = {
        "configurable": {
            "thread_id": "1",
            "model_type": "haiku",  # Example model type
            "retriever_path": "GP_Copilot/chroma",  # Example retriever
            "custom_system_prompt": "",
            "verbose": True
        }
    }
    
    initial_state = {
        "session_id": "1",
        "messages": [{"role": "user", "content": 'hello'}]
    }

    graph = get_graph()
    
    print(graph.invoke(initial_state, config)['messages'][-1].content)


