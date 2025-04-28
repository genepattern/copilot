from datetime import datetime
from dotenv import load_dotenv
from django.conf import settings
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import START, StateGraph, END, MessagesState
from typing import List
from .models import LlmModel, SystemPrompt, Conversation, Query, Step


_instance = None


class ServiceHelper:
    """Helper class to manage LLM services and singleton instance"""

    @staticmethod
    def load_llms():
        """Load all LLM models from the database and initialize them."""
        llms = {}
        for model in LlmModel.objects.filter(disabled=False):
            llms[model.model_id] = init_chat_model(model.model_id, model_provider=model.provider_id, temperature=0.1)
        return llms

    def load_vector_store(self):
        """Load the vector store and embeddings"""
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = Chroma(
            collection_name="moduledoc",
            embedding_function=embeddings,
            persist_directory="./vectorstore/chroma",
        )
        return vector_store

    def __init__(self):
        load_dotenv()                                   # Load environment variables (especially API keys)
        self.llms = self.load_llms()                    # Initialize LLM models
        self.vector_store = self.load_vector_store()    # Load vector store


def instance():
    """Singleton instance for LLM services"""
    global _instance
    if _instance is None: _instance = ServiceHelper()
    return _instance


class ConversationState(MessagesState):
    """Defines the state passed between nodes in the graph."""
    conversation_id: str
    model_id: str
    prompt: str
    query: str
    context: List
    answer: str
    steps: List


def retrieve_documents(state: ConversationState):
    """Retrieve relevant documents from the vector store based on the query"""
    started_at = datetime.now()
    docs = instance().vector_store.similarity_search(state["query"])
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


def answer_question(state: ConversationState):
    """Answer the question using the retrieved documents and the LLM"""
    model_id = state["model_id"]
    if model_id not in instance().llms:
        raise ValueError(f"Model '{model_id}' not found in loaded LLM models.")

    context = "\n\n".join(doc.page_content for doc in state["context"])
    system = SystemMessage(content=(state["prompt"] + "\n\n" + context + "\n\n"))

    history = [message for message in state["messages"] if message.type in ("human", "ai")]
    full_prompt = [system] + history + [HumanMessage(content=("\n\n" + state["query"]))]

    started_at = datetime.now()
    response = instance().llms[model_id].invoke(full_prompt)
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


def build_langgraph():
    """Build and compile the LangGraph for handling conversations"""
    workflow = StateGraph(ConversationState)

    # Add nodes
    workflow.add_node("retrieve_documents", retrieve_documents)
    workflow.add_node("answer_question", answer_question)

    # Define edges
    workflow.add_edge(START, "retrieve_documents")
    workflow.add_edge("retrieve_documents", "answer_question")
    workflow.add_edge("answer_question", END)

    # Compile the graph
    app = workflow.compile()
    return app


# Compile graph once, when module loads
graph = build_langgraph()


def assemble_answer(answer):
    if isinstance(answer, str): return answer
    if isinstance(answer, tuple) or isinstance(answer, str):
        if all(isinstance(item, str) for item in answer):
            return '\n\n'.join(answer)
        if all(isinstance(item, list) for item in answer) and len(answer):  # Special case for DeepSeek
            for item in answer[0]:
                if 'text' in item: return item['text']
    raise ValueError("Invalid answer format. Expected a string, tuple or list of strings")


def handle_chat_message(user, conversation_id, user_query, model_id=None, system_prompt_id=None):
    """ Handles an incoming chat message"""

    start_time = datetime.now()         # Note start time
    if user.is_anonymous: user = None   # Anonymous users should be null

    # 1. Get the existing conversation or lazily create one
    if conversation_id:
        try: conversation = Conversation.objects.get(id=conversation_id)
        except Conversation.DoesNotExist: return None, "Conversation not found or access denied"
    else:
        conversation = Conversation.objects.create(user=user)
        conversation_id = conversation.id  # Get the new ID

    # 2. Select LLM Model
    if model_id:
        try: llm_model = LlmModel.objects.get(model_id=model_id)
        except LlmModel.DoesNotExist: return None, "Requested model id not found"
    else:
        model_id = settings.DEFAULT_LLM_MODEL
        try: llm_model = LlmModel.objects.get(model_id=model_id)
        except LlmModel.DoesNotExist: return None, "Requested model id not found"

    # Handle case where *no* models are found
    if not llm_model: return None, "No suitable model found or configured."
    model_id = llm_model.model_id

    # 3. Select System Prompt
    if system_prompt_id:  # TODO: Handle requesting specific version or (id vs name)
        try: system_prompt = SystemPrompt.objects.filter(name=system_prompt_id).order_by('-version').first()
        except SystemPrompt.DoesNotExist: return None, "Requested system prompt not found"
    else: system_prompt = SystemPrompt.objects.all().first()  # Default to first model  # TODO: Better default handling

    # Handle case where *no* system prompts are found
    if not system_prompt: return None, "No suitable model found or configured."

    # 4. Prepare Initial State for LangGraph
    initial_state = ConversationState(
        conversation_id=conversation.id,
        model_id=model_id,
        prompt=system_prompt.prompt,
        query=user_query,
        steps=[],
        messages=[],
        context=[],
        answer=""
    )

    # # 5. Run the LangGraph
    final_state = graph.invoke(initial_state)

    # 6. Record Query and Steps in Database
    end_time = datetime.now()
    query_num = conversation.queries.count() + 1
    answer = final_state.get('answer', "Error: No response generated."),  # "I can't let you do that, Dave."
    answer = assemble_answer(answer)

    query_instance = Query.objects.create(conversation=conversation, query_num=query_num, llm_model=llm_model,
                                          started_at=start_time, ended_at=end_time, raw_query=user_query, response=answer)

    # # Save steps taken during the graph execution
    for i, step in enumerate(final_state.get('steps', [])):
        Step.objects.create(
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
