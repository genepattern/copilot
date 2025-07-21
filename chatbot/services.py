import asyncio
from datetime import datetime
from asgiref.sync import sync_to_async
from dotenv import load_dotenv
from django.conf import settings
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import START, StateGraph, END, MessagesState
from typing import List, Dict
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from .models import LlmModel, SystemPrompt, Conversation, Query, Step
import logging
import threading

logger = logging.getLogger(__name__)

# Global variables for caching expensive-to-load, immutable resources, initialized when first imported
_cached_llms = None
_cached_vector_store = None
_cached_tools = None
_cached_graphs = None

# A simple, synchronous lock for global cache initialization, prevents race conditions
_cache_lock = threading.Lock()


class ServiceHelper:
    """
        Initializes the ServiceHelper.
        Args:
            llms (dict): Dictionary of initialized LLM models.
            vector_store: Initialized vector store instance.
            graphs (dict): Dictionary of pre-built LangGraph instances.
            tools (list): List of available tools for the LLM.
        """

    def __init__(self, llms, vector_store, graphs, tools):
        self.llms = llms
        self.vector_store = vector_store
        self._graph: Dict[str, StateGraph] = graphs
        self.tools = tools

    def graph(self, method: str) -> StateGraph:
        graph_instance = self._graph.get(method)
        if not graph_instance:
            valid_methods = list(self._graph.keys())
            raise ValueError(f"Invalid method '{method}'. Valid methods are: {valid_methods}")
        return graph_instance

    @staticmethod
    async def async_orm_wrapper(func, **kwargs):
        """Asynchronous ORM wrapper to call a function with kwargs."""
        return await sync_to_async(
            lambda: func(**kwargs),
            thread_sensitive=True
        )()

    @staticmethod
    async def async_orm_create(model_cls, **kwargs):
        """Asynchronous ORM create method."""
        return await ServiceHelper.async_orm_wrapper(model_cls.objects.create, **kwargs)

    @staticmethod
    async def async_orm_get(model_cls, **kwargs):
        """Asynchronous ORM get method to fetch a single object."""
        return await ServiceHelper.async_orm_wrapper(model_cls.objects.get, **kwargs)

    @staticmethod
    async def async_orm_filter(model_cls, **kwargs):
        """Asynchronous ORM filter method."""
        return await ServiceHelper.async_orm_wrapper(model_cls.objects.filter, **kwargs)

    @staticmethod
    async def async_orm_filter_sort_first(model_cls, sort, **kwargs):
        """Fetch the first object matching the filter and sort criteria."""
        return await ServiceHelper.async_orm_wrapper(
            lambda **x: model_cls.objects.filter(**kwargs).order_by(sort).first(), **kwargs
        )

    @staticmethod
    @sync_to_async
    def get_enabled_llms():
        """Retrieves a list of enabled LLM models from the database."""
        return list(LlmModel.objects.filter(disabled=False))

    @staticmethod
    async def _load_llms():
        """Load all enabled LLM models from the database and initialize them."""
        llms = {}
        models = await ServiceHelper.get_enabled_llms()
        for model in models:
            llms[model.model_id] = init_chat_model(
                model.model_id, model_provider=model.provider_id, temperature=0.1
            )
        return llms

    @staticmethod
    def _load_vector_store():
        """Load the vector store and embeddings."""
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return Chroma(
            collection_name="moduledoc",
            embedding_function=embeddings,
            persist_directory="./vectorstore/chroma",
        )

    @staticmethod
    async def _load_mcp_tools():
        """Load the GenePattern MCP client and its tools."""
        try:
            mcp_url = getattr(settings, 'GENEPATTERN_MCP_URL', "http://localhost:3000/mcp")
            client = MultiServerMCPClient({
                "genepattern": {"transport": "streamable_http", "url": mcp_url},
            })
            return await client.get_tools()
        except Exception as e:
            logger.error(f"Could not connect to MCP server or load tools: {e}", exc_info=True)
            return []

    @classmethod
    async def create_instance(cls):
        """Asynchronously creates and configures a new ServiceHelper instance,
        using globally cached resources if available.
        """
        global _cached_llms, _cached_vector_store, _cached_tools, _cached_graphs

        # Use a synchronous lock to guard the global cache initialization block.
        with _cache_lock:
            load_dotenv()

            # Lazy load and cache LLMs
            if _cached_llms is None:
                logger.info("Initializing and caching LLMs...")
                _cached_llms = await cls._load_llms()
            llms = _cached_llms

            # Lazy load and cache Vector Store
            if _cached_vector_store is None:
                logger.info("Initializing and caching vector store...")
                _cached_vector_store = cls._load_vector_store()
            vector_store = _cached_vector_store

            # Lazy load and cache MCP Tools
            if _cached_tools is None:
                logger.info("Initializing and caching MCP tools...")
                _cached_tools = await cls._load_mcp_tools()
            tools = _cached_tools

            # Lazy build and cache LangGraphs
            if _cached_graphs is None:
                logger.info("Building and caching LangGraphs...")
                try:
                    _cached_graphs = await build_langgraph(tools)
                except Exception as e:
                    logger.error(f"Error building LangGraphs during caching: {e}", exc_info=True)
                    raise
            graphs = _cached_graphs

        return cls(llms=llms, vector_store=vector_store, graphs=graphs, tools=tools)


class ConversationState(MessagesState):
    """Defines the state passed between nodes in the graph."""
    conversation_id: str
    model_id: str
    prompt: str
    raw_query: str
    query: str
    context: List
    answer: str
    steps: List
    api_key: str = None


async def genepattern_mcp(state: ConversationState):
    """Node to interact with the LLM and GenePattern tools via MCP."""
    logger.debug("\n--- Entering genepattern_mcp ---")
    started_at = datetime.now()
    helper = await ServiceHelper.create_instance()
    model_id = state["model_id"]

    if model_id not in helper.llms:
        raise ValueError(f"Model '{model_id}' not found in loaded LLM models.")

    # Create the initial message if the history is empty
    if not state["messages"]:
        state["messages"] = [HumanMessage(content=state["query"])]

    # For accurate logging, capture the messages that are being sent to the LLM
    step_input_messages = str(state["messages"])

    # Invoke the LLM
    model_with_tools = helper.llms[model_id].bind_tools(helper.tools)
    response = await model_with_tools.ainvoke(state["messages"])
    ended_at = datetime.now()

    # Log the details of this step
    state["steps"].append({
        'llm_model': state["model_id"],
        'system_prompt': state["prompt"],
        'call_id': 'genepattern_mcp',
        'step_input': step_input_messages,
        'step_output': str(response),
        'started_at': started_at,
        'ended_at': ended_at,
    })

    state["messages"].append(response)

    # Log whether a tool call was requested
    if response.tool_calls:
        logger.info(f"LLM wants to call a tool: {response.tool_calls[0].get('name')}")
    else:
        logger.info("LLM did not request a tool call, proceeding to answer.")

    # Return the updated state fields
    return {"messages": state["messages"], "steps": state["steps"]}


async def retrieve_documents(state: ConversationState):
    """Node to retrieve relevant documents from the vector store."""
    started_at = datetime.now()
    helper = await ServiceHelper.create_instance()
    docs = helper.vector_store.similarity_search(state["query"])
    ended_at = datetime.now()
    state["steps"].append({
        'llm_model': state["model_id"],
        'system_prompt': state["prompt"],
        'call_id': 'retrieve_documents[all]',
        'step_input': state["prompt"],
        'step_output': "\n\n".join(doc.page_content for doc in docs),
        'started_at': started_at,
        'ended_at': ended_at,
    })
    return { "context": docs }


async def answer_question(state: ConversationState):
    """
    Node to generate a final answer using the LLM and context.
    This node is designed to handle being called from different graph paths.
    """
    model_id = state["model_id"]
    helper = await ServiceHelper.create_instance() # Get a ServiceHelper instance with cached resources
    if model_id not in helper.llms:
        raise ValueError(f"Model '{model_id}' not found in loaded LLM models.")

    # Add API key to context if provided
    if state.get('method_id') == 'mcp' and state.get("api_key"):
        context += f"\n\nGenePattern API Key: {state['api_key']}"

    context = "\n\n".join(doc.page_content for doc in state.get("context", []))
    system_content = f"{state['prompt']}\n\n{context}\n\n"
    system = SystemMessage(content=system_content)

    history = state.get("messages", [])

    has_tool_messages = any(
        (isinstance(msg, AIMessage) and msg.tool_calls) or isinstance(msg, ToolMessage)
        for msg in history
    )

    if has_tool_messages or state.get('method_id') == 'mcp':
        llm_to_use = helper.llms[model_id].bind_tools(helper.tools)
    else:
        llm_to_use = helper.llms[model_id]

    if not history:
        full_prompt = [system, HumanMessage(content=state['query'])]
    else:
        full_prompt = [system] + history
        # Ensure the conversation ends with a HumanMessage for models requiring it.
        if not full_prompt or not isinstance(full_prompt[-1], HumanMessage):
            full_prompt.append(HumanMessage(content=f"Based on the conversation above, please provide the final answer to my original question: {state['raw_query']}"))

    started_at = datetime.now()
    response = await llm_to_use.ainvoke(full_prompt)
    ended_at = datetime.now()
    state["steps"].append({
        'llm_model': state["model_id"],
        'system_prompt': state["prompt"],
        'call_id': 'answer_question',
        'step_input': str(full_prompt),
        'step_output': response.content,
        'started_at': started_at,
        'ended_at': ended_at,
    })
    return {"answer": response.content}


async def summarize_question_for_raw_graph(state: ConversationState):
    """
    A specific async node for the raw graph to call summarize_question
    with llm_summarization=False and await its result.
    """
    return await summarize_question(state, llm_summarization=False)


async def summarize_question(state: ConversationState, llm_summarization=True):
    """Node to summarize the user's raw query."""

    # Check if LLM summarization is enabled
    if not llm_summarization: return {"query": state["raw_query"]}

    model_id = state["model_id"]
    helper = await ServiceHelper.create_instance()
    if model_id not in helper.llms:
        raise ValueError(f"Model '{model_id}' not found in loaded LLM models.")

    started_at = datetime.now()
    system_prompt = await ServiceHelper.async_orm_get(SystemPrompt, name="Summarize Question", version=1.0)
    system = SystemMessage(content=f"{system_prompt.prompt}\n\n")
    full_prompt = [system, HumanMessage(content=state["raw_query"])]
    response = await helper.llms[model_id].ainvoke(full_prompt)
    ended_at = datetime.now()

    state["steps"].append({
        'llm_model': state["model_id"],
        'system_prompt': system_prompt.prompt,
        'call_id': 'summarize_question',
        'step_input': state["raw_query"],
        'step_output': response.content,
        'started_at': started_at,
        'ended_at': ended_at,
    })

    return { "query": response.content }


async def build_rag_graph() -> StateGraph:
    """Build and compile the LangGraph for RAG."""
    workflow = StateGraph(ConversationState)

    # Add nodes
    workflow.add_node("summarize_question", summarize_question)
    workflow.add_node("retrieve_documents", retrieve_documents)
    workflow.add_node("answer_question", answer_question)

    # Define edges
    workflow.add_edge(START, "summarize_question")
    workflow.add_edge("summarize_question", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "answer_question")
    workflow.add_edge("answer_question", END)

    # Compile the graph
    return workflow.compile()


async def build_mcp_graph(tools) -> StateGraph:
    """Build and compile the LangGraph for MCP tool usage."""
    workflow = StateGraph(ConversationState)

    # 1. Add all nodes to the graph
    workflow.add_node("summarize_question", summarize_question)
    workflow.add_node("genepattern_mcp", genepattern_mcp)
    workflow.add_node("answer_question", answer_question)

    # The ToolNode is a pre-built node that executes the tools it's given
    workflow.add_node("tools", ToolNode(tools))

    # 2. Define the graph's flow (edges)
    workflow.add_edge(START, "summarize_question")
    workflow.add_edge("summarize_question", "genepattern_mcp")

    # 3. Add the conditional edge for tool calling
    # After the `genepattern_mcp` node runs, the `tools_condition` function checks
    # if the last message contains tool calls.
    workflow.add_conditional_edges(
        "genepattern_mcp",
        tools_condition,
        # If `tools_condition` is TRUE, it routes to the "tools" node.
        # If `tools_condition` is FALSE, it routes to the "answer_question" node.
        {"tools": "tools", "__end__": "answer_question"},
    )

    # 4. Define the loop
    # After the "tools" node runs, it loops back to the `genepattern_mcp` node
    # so the LLM can process the tool results.
    workflow.add_edge("tools", "genepattern_mcp")

    # 5. Define the final step
    workflow.add_edge("answer_question", END)

    return workflow.compile()


async def build_raw_graph() -> StateGraph:
    """Build and compile the LangGraph for direct answering."""
    workflow = StateGraph(ConversationState)

    # Use the specific async helper node that awaits summarize_question
    workflow.add_node("summarize_question", summarize_question_for_raw_graph)
    workflow.add_node("answer_question", answer_question)
    workflow.add_edge(START, "summarize_question")
    workflow.add_edge("summarize_question", "answer_question")
    workflow.add_edge("answer_question", END)

    return workflow.compile()


async def build_langgraph(tools) -> Dict[str, StateGraph]:
    """Build and compile all LangGraphs and return them in a dictionary."""
    rag_graph, mcp_graph, raw_graph = await asyncio.gather(
        build_rag_graph(),
        build_mcp_graph(tools),
        build_raw_graph()
    )
    return { 'rag': rag_graph, 'mcp': mcp_graph, 'raw': raw_graph }


def assemble_answer(answer):
    """Assembles the final answer from potentially complex LLM outputs."""
    if isinstance(answer, str): return answer
    if isinstance(answer, tuple) or isinstance(answer, str):
        if all(isinstance(item, str) for item in answer):
            return '\n\n'.join(answer)
        if all(isinstance(item, list) for item in answer) and len(answer):  # Special case for DeepSeek
            for item in answer[0]:
                if 'text' in item: return item['text']
    raise ValueError("Invalid answer format. Expected a string, tuple or list of strings")


async def handle_chat_message(user, conversation_id, user_query, model_id=None, method_id=None, system_prompt_id=None, api_key=None):
    """Handles an incoming chat message, runs it through the appropriate graph and logs the results."""

    start_time = datetime.now()
    if user.is_anonymous: user = None  # Anonymous users should be null

    # 1. Get the existing conversation or lazily create one
    if conversation_id:
        try: conversation = await ServiceHelper.async_orm_get(Conversation, id=conversation_id)
        except Conversation.DoesNotExist: return None, "Conversation not found or access denied"
    else:
        conversation = await ServiceHelper.async_orm_create(Conversation, user=user)
        conversation_id = conversation.id  # Get the new ID

    # 2. Select LLM Model
    if not model_id: model_id = settings.DEFAULT_LLM_MODEL
    try: llm_model = await ServiceHelper.async_orm_get(LlmModel, model_id=model_id)
    except LlmModel.DoesNotExist: return None, "Requested model id not found"

    # Handle case where *no* models are found
    if not llm_model: return None, "No suitable model found or configured."
    model_id = llm_model.model_id

    # 2.5 Select LLM Method
    if not method_id: method_id = settings.DEFAULT_LLM_METHOD

    # 3. Select System Prompt
    if system_prompt_id:  # TODO: Handle requesting specific version or (id vs name)
        try: system_prompt = await ServiceHelper.async_orm_filter_sort_first(SystemPrompt, '-version', name=system_prompt_id)
        except SystemPrompt.DoesNotExist: return None, "Requested system prompt not found"
    else: system_prompt = await ServiceHelper.async_orm_filter_sort_first(SystemPrompt, '-version', name="General")

    # Handle case where *no* system prompts are found
    if not system_prompt: return None, "No suitable model found or configured."

    # 4. Prepare Initial State for LangGraph
    initial_state = ConversationState(
        conversation_id=conversation.id,
        model_id=model_id,
        prompt=system_prompt.prompt,
        raw_query=user_query,
        query="",
        steps=[],
        messages=[],
        context=[],
        answer="",
        api_key=api_key
    )

    # 5. Run the LangGraph
    helper = await ServiceHelper.create_instance()
    final_state = await helper.graph(method_id).ainvoke(initial_state)

    # 6. Record Query and Steps in Database
    end_time = datetime.now()
    query_num = await sync_to_async(conversation.queries.count)() + 1
    answer = final_state.get('answer', "Error: No response generated."),
    answer = assemble_answer(answer)

    query_instance = await ServiceHelper.async_orm_create(Query, conversation=conversation, query_num=query_num,
                                                          llm_model=llm_model, started_at=start_time, ended_at=end_time,
                                                          raw_query=user_query, response=answer)

    # Save steps taken during the graph execution
    for i, step in enumerate(final_state.get('steps', [])):
        await ServiceHelper.async_orm_create(
            Step,
            query=query_instance,
            step_num=i + 1,
            llm_model=llm_model,
            system_prompt=system_prompt,
            call_id=str(step["call_id"]),
            step_input=str(step["step_input"]),
            step_output=str(step["step_output"]),
            started_at=step["started_at"],
            ended_at=step["ended_at"]
        )

    # 7. Return the created Query object
    return query_instance, None
