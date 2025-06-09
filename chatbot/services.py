import asyncio
from datetime import datetime
from asgiref.sync import sync_to_async
from dotenv import load_dotenv
from django.conf import settings
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import START, StateGraph, END, MessagesState
from typing import List
from langgraph.prebuilt import ToolNode, tools_condition
from .models import LlmModel, SystemPrompt, Conversation, Query, Step
import logging

logger = logging.getLogger(__name__) # Use __name__ for module-specific logger


_instance = None
_instance_lock = asyncio.Lock()


class ServiceHelper:
    """Helper class to manage LLM services and singleton instance"""

    def __init__(self, llms=None, vector_store=None, graph=None, tools=None):
        self.llms = llms or {}
        self.vector_store = vector_store
        self.graph = graph
        self.tools = tools or []

    @staticmethod
    async def async_orm_wrapper(func, **kwargs):
        """Asynchronous ORM wrapper to call a function with kwargs"""
        result = await sync_to_async(
            lambda: func(**kwargs),
            thread_sensitive=True  # set False if it's thread-safe
        )()
        return result

    @staticmethod
    async def async_orm_create(model_cls, **kwargs):
        """Asynchronous ORM get method to fetch a single object"""
        return await ServiceHelper.async_orm_wrapper(model_cls.objects.create, **kwargs)

    @staticmethod
    async def async_orm_get(model_cls, **kwargs):
        """Asynchronous ORM get method to fetch a single object"""
        return await ServiceHelper.async_orm_wrapper(model_cls.objects.get, **kwargs)

    @staticmethod
    async def async_orm_filter(model_cls, **kwargs):
        """Asynchronous ORM get method to fetch a single object"""
        return await ServiceHelper.async_orm_wrapper(model_cls.objects.filter, **kwargs)

    @staticmethod
    async def async_orm_filter_sort_first(model_cls, sort, **kwargs):
        """Fetch the first object matching the filter and sort criteria"""
        return await ServiceHelper.async_orm_wrapper(lambda **x: model_cls.objects.filter(**kwargs).order_by(sort).first(), **kwargs)

    @staticmethod
    @sync_to_async
    def get_enabled_llms():
        return list(LlmModel.objects.filter(disabled=False))

    @staticmethod
    async def load_llms():
        """Load all LLM models from the database and initialize them."""
        llms = {}
        models = await ServiceHelper.get_enabled_llms()
        for model in models:
            llms[model.model_id] = init_chat_model(model.model_id, model_provider=model.provider_id, temperature=0.1)
        return llms

    @staticmethod
    def load_vector_store():
        """Load the vector store and embeddings"""
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = Chroma(
            collection_name="moduledoc",
            embedding_function=embeddings,
            persist_directory="./vectorstore/chroma",
        )
        return vector_store

    @staticmethod
    async def load_mcp_tools():
        """Load the GenePattern MCP client"""
        try:
            mcp_url = getattr(settings, 'GENEPATTERN_MCP_URL', "http://localhost:3000/mcp")
            client = MultiServerMCPClient({
                "genepattern": {"transport": "streamable_http", "url": mcp_url},
            })
            tools = await client.get_tools()
            return tools
        except Exception as e:
            logger.error(f"Could not connect to MCP server or load tools: {e}", exc_info=True)
            return []

    async def create(self):
        load_dotenv()
        self.llms = await self.load_llms()
        self.vector_store = self.load_vector_store()
        self.tools = await self.load_mcp_tools()
        try: self.graph = await build_langgraph(self)  # Pass 'self' (the ServiceHelper instance)
        except Exception as e:
            logger.error(f"Error building LangGraph: {e}", exc_info=True)
            raise
        return self


async def instance():
    """Singleton instance for LLM services"""
    global _instance
    async with _instance_lock:
        if _instance is None: _instance = await ServiceHelper().create()
    return _instance


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


async def genepattern_mcp(state: ConversationState):
    logger.debug("\n--- Entering genepattern_mcp ---")
    started_at = datetime.now()
    helper = await instance()
    model_id = state["model_id"]

    if model_id not in helper.llms:
        raise ValueError(f"Model '{model_id}' not found in loaded LLM models.")

    if not state["messages"]:
        state["messages"] = [HumanMessage(content=state["query"])]

    model_with_tools = helper.llms[model_id].bind_tools(helper.tools)
    response = await model_with_tools.ainvoke(state["messages"])
    ended_at = datetime.now()

    state["steps"].append({
        'llm_model': state["model_id"],
        'system_prompt': state["prompt"],
        'call_id': 'genepattern_mcp',
        'step_input': str(state["messages"]),
        'step_output': str(response),
        'started_at': started_at,
        'ended_at': ended_at,
    })
    state["messages"].append(response)

    if response.tool_calls:
        first_tool_call = response.tool_calls[0]
        tool_name = first_tool_call.get('name')
        logger.info(f"LLM wants to call a tool: {tool_name}")

    return {"messages": state["messages"], "steps": state["steps"]}


async def retrieve_documents(state: ConversationState):
    """Retrieve relevant documents from the vector store based on the query"""
    started_at = datetime.now()
    helper = await instance()  # Get the singleton instance of ServiceHelper
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
    """Answer the question using the retrieved documents and the LLM"""
    model_id = state["model_id"]
    helper = await instance()  # Get the singleton instance of ServiceHelper
    if model_id not in helper.llms:
        raise ValueError(f"Model '{model_id}' not found in loaded LLM models.")

    context = "\n\n".join(doc.page_content for doc in state["context"])
    system = SystemMessage(content=(state["prompt"] + "\n\n" + context + "\n\n"))

    history = [message for message in state["messages"] if message.type in ("human", "ai")]
    full_prompt = [system] + history + [HumanMessage(content=("\n\n" + state["query"]))]

    started_at = datetime.now()
    response = await helper.llms[model_id].ainvoke(full_prompt)
    ended_at = datetime.now()
    state["steps"].append({
        'llm_model': state["model_id"],
        'system_prompt': state["prompt"],
        'call_id': 'answer_question',
        'step_input': state["prompt"],
        'step_output': response.content,
        'started_at': started_at,
        'ended_at': ended_at,
    })
    return { "messages": response, "answer": response.content }


async def summarize_question(state: ConversationState, llm_summarization=True):
    """Summarize the question asked by the user"""

    # Check if LLM summarization is enabled
    if not llm_summarization: return { "query": state["raw_query"] }

    model_id = state["model_id"]
    helper = await instance()  # Get the singleton instance of ServiceHelper
    if model_id not in helper.llms:
        raise ValueError(f"Model '{model_id}' not found in loaded LLM models.")

    started_at = datetime.now()
    system_prompt = await ServiceHelper.async_orm_get(SystemPrompt, name="Summarize Question", version=1.0)
    system = SystemMessage(content=(system_prompt.prompt + '\n\n'))
    full_prompt = [system] + [HumanMessage(content=(state["raw_query"]))]
    response = await helper.llms[model_id].ainvoke(full_prompt)
    state["query"] = response.content
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


async def build_rag_graph():
    """Build and compile the LangGraph for handling conversations with RAG"""
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
    app = workflow.compile()
    return app


async def build_mcp_graph(helper_instance):
    """Build and compile the LangGraph for handling conversations with MCP"""
    workflow = StateGraph(ConversationState)

    workflow.add_node("summarize_question", summarize_question)
    workflow.add_node("genepattern_mcp", genepattern_mcp)
    workflow.add_node("answer_question", answer_question)
    workflow.add_node("tools", ToolNode(helper_instance.tools))

    workflow.add_edge(START, "summarize_question")
    workflow.add_edge("summarize_question", "genepattern_mcp")
    workflow.add_conditional_edges("genepattern_mcp", tools_condition,
                                   { "tools": "tools", "__end__": "answer_question" }, )
    workflow.add_edge("tools", "genepattern_mcp")
    workflow.add_edge("answer_question", END)

    app = workflow.compile()
    return app


async def build_langgraph(helper_instance, rag=False):
    """Build and compile the LangGraph for handling conversations"""
    if rag: return await build_rag_graph()
    else: return await build_mcp_graph(helper_instance)


def assemble_answer(answer):
    if isinstance(answer, str): return answer
    if isinstance(answer, tuple) or isinstance(answer, str):
        if all(isinstance(item, str) for item in answer):
            return '\n\n'.join(answer)
        if all(isinstance(item, list) for item in answer) and len(answer):  # Special case for DeepSeek
            for item in answer[0]:
                if 'text' in item: return item['text']
    raise ValueError("Invalid answer format. Expected a string, tuple or list of strings")


async def handle_chat_message(user, conversation_id, user_query, model_id=None, system_prompt_id=None):
    """ Handles an incoming chat message"""

    start_time = datetime.now()         # Note start time
    if user.is_anonymous: user = None   # Anonymous users should be null

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
        answer=""
    )

    # # 5. Run the LangGraph
    helper = await instance()  # Get the singleton instance of ServiceHelper
    final_state = await helper.graph.ainvoke(initial_state)

    # 6. Record Query and Steps in Database
    end_time = datetime.now()
    query_num = await sync_to_async(conversation.queries.count)() + 1
    answer = final_state.get('answer', "Error: No response generated."),
    answer = assemble_answer(answer)

    query_instance = await ServiceHelper.async_orm_create(Query, conversation=conversation, query_num=query_num,
                                                          llm_model=llm_model, started_at=start_time, ended_at=end_time,
                                                          raw_query=user_query, response=answer)  #Query.objects.create(conversation=conversation, query_num=query_num, llm_model=llm_model, started_at=start_time, ended_at=end_time, raw_query=user_query, response=answer)

    # # Save steps taken during the graph execution
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
    return query_instance, None  # Return query instance and no error message