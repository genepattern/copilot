from dotenv import load_dotenv
from django.conf import settings
from django.utils import timezone
from django.db.models import Sum
from django.core.mail import send_mail
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import chromadb
from chromadb.utils import embedding_functions
import logging
import threading
import base64
import mimetypes
import httpx

from .models import LlmModel, SystemPrompt, Conversation, Query, Step, TokenCount

logger = logging.getLogger(__name__)

# Global cache for expensive resources
_cached_llms = None
_cached_vector_store = None
_cached_tools = None
_cache_lock = threading.Lock()

# Email rate limiting
_last_token_limit_email_sent = None
_email_lock = threading.Lock()


@dataclass
class ConversationState:
    """State passed between agent runs."""
    conversation_id: str
    model_id: str
    prompt: str
    raw_query: str
    query: str = ""
    context: List = field(default_factory=list)
    answer: str = ""
    steps: List = field(default_factory=list)
    method_id: Optional[str] = None
    api_key: Optional[str] = None
    files: Optional[List] = None
    files_content: Optional[str] = None
    conversation_history: str = ""  # Add conversation history field


class ServiceHelper:
    """Helper class for managing LLMs, vector store, and MCP tools."""

    def __init__(self, llms, vector_store, tools):
        self.llms = llms
        self.vector_store = vector_store
        self.tools = tools

    @staticmethod
    def _load_llms():
        """Load all enabled LLM models from the database."""
        llms = {}
        models = LlmModel.objects.filter(disabled=False)

        for model in models:
            if model.provider_id in ('google_genai', 'google-gla'):
                model_name = model.model_id
            else:
                provider = 'bedrock' if model.provider_id == 'bedrock_converse' else model.provider_id
                model_name = f"{provider}:{model.model_id}"

            llms[model.model_id] = model_name
            logger.info(f"Loaded model {model.model_id} as '{model_name}'")

        return llms

    @staticmethod
    def _load_vector_store():
        """Load the ChromaDB vector store."""
        chroma_client = chromadb.PersistentClient(path="./vectorstore/chroma")
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        return chroma_client.get_or_create_collection(
            name="moduledoc",
            embedding_function=embedding_function
        )

    @staticmethod
    def _load_mcp_server(api_key: Optional[str] = None):
        """Initialize MCP server connection using Pydantic AI's native MCP support."""
        mcp_url = getattr(settings, 'GENEPATTERN_MCP_URL', "http://localhost:3000/mcp")

        logger.info(f"="*70)
        logger.info(f"Initializing MCP server connection at: {mcp_url}")
        logger.info(f"API key provided: {api_key is not None}")
        if api_key:
            logger.info(f"API key length: {len(api_key)}")
        logger.info(f"="*70)

        try:
            # Create MCP server with optional authorization header
            if api_key:
                # Pass headers directly to MCPServerStreamableHTTP
                # The MCP transport will use these headers for ALL requests including tool calls
                mcp_server = MCPServerStreamableHTTP(
                    mcp_url,
                    headers={"Authorization": f"Bearer {api_key}"}
                )
                logger.info(f"✓ MCP server initialized with API key authentication")
                logger.info(f"  Authorization header: Bearer {api_key[:8]}...")
            else:
                mcp_server = MCPServerStreamableHTTP(mcp_url)
                logger.info(f"✓ MCP server initialized without authentication")

            return mcp_server

        except Exception as e:
            logger.error(f"="*70)
            logger.error(f"✗ MCP INITIALIZATION ERROR")
            logger.error(f"="*70)
            logger.error(f"Failed to initialize MCP server at {mcp_url}")
            logger.error(f"Error: {type(e).__name__}: {str(e)}")
            logger.error(f"")
            logger.error(f"TROUBLESHOOTING:")
            logger.error(f"  1. Verify MCP server is running: curl {mcp_url}")
            logger.error(f"  2. Check that server supports streamable-http transport")
            logger.error(f"  3. Verify URL in settings.GENEPATTERN_MCP_URL")
            logger.error(f"  4. Check server logs for errors")
            logger.error(f"")
            logger.error(f"Application will continue without MCP tools.")
            logger.error(f"="*70)
            import traceback
            logger.error(traceback.format_exc())
            return None

    @classmethod
    def create_instance(cls, api_key: Optional[str] = None):
        """Create a ServiceHelper instance with cached or per-request resources."""
        global _cached_llms, _cached_vector_store, _cached_tools

        with _cache_lock:
            load_dotenv()

            # Cache LLMs and vector store (shared across requests)
            if _cached_llms is None:
                logger.info("Initializing and caching LLMs...")
                _cached_llms = cls._load_llms()

            if _cached_vector_store is None:
                logger.info("Initializing and caching vector store...")
                _cached_vector_store = cls._load_vector_store()

            # MCP tools: use per-request if api_key provided, otherwise cache
            if api_key:
                logger.debug("Loading MCP tools with per-request Authorization header")
                tools = cls._load_mcp_server(api_key)
            else:
                if _cached_tools is None:
                    logger.info("Initializing and caching MCP tools...")
                    _cached_tools = cls._load_mcp_server()
                tools = _cached_tools

        return cls(llms=_cached_llms, vector_store=_cached_vector_store, tools=tools)


# ==================== Helper Functions ====================

def read_and_format_files(files):
    """Read and format uploaded files for inclusion in prompts."""
    if not files:
        return ""

    files_content = "\n\n### Attached Files:\n\n"
    for file in files:
        try:
            content = file.read()
            try:
                text_content = content.decode('utf-8')
                files_content += f"**File: {file.name}**\n```\n{text_content}\n```\n\n"
            except UnicodeDecodeError:
                b64_content = base64.b64encode(content).decode('ascii')
                mimetype, _ = mimetypes.guess_type(file.name)
                mimetype = mimetype or 'application/octet-stream'
                files_content += (
                    f"**File: {file.name}** (binary, {len(content)} bytes)\n"
                    f"MIME type: {mimetype}\n"
                    f"Base64: {b64_content}\n\n"
                )
        except Exception as e:
            logger.error(f"Error reading file {file.name}: {e}")
            files_content += f"**File: {file.name}** (error reading file)\n\n"

    return files_content


def estimate_token_counts(query: str, result_text: str) -> Dict[str, Any]:
    """Estimate token usage based on word count."""
    prompt_tokens = len(query.split()) * 2
    completion_tokens = len(result_text.split()) * 2
    return {
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'total_tokens': prompt_tokens + completion_tokens,
        'estimated': True
    }


def extract_result_text(result) -> str:
    """Extract text from Pydantic AI result object."""
    if hasattr(result, 'output'):
        return str(result.output)
    elif hasattr(result, 'data'):
        return str(result.data)
    else:
        return str(result)


def retrieve_documents(state: ConversationState, helper: ServiceHelper):
    """Retrieve relevant documents from the vector store."""
    started_at = timezone.now()

    results = helper.vector_store.query(query_texts=[state.query], n_results=5)

    docs = []
    if results and results.get('documents'):
        for doc_list in results['documents']:
            docs.extend(doc_list)

    ended_at = timezone.now()
    state.steps.append({
        'llm_model': state.model_id,
        'system_prompt': state.prompt,
        'call_id': 'retrieve_documents',
        'step_input': state.query,
        'step_output': "\n\n".join(docs),
        'started_at': started_at,
        'ended_at': ended_at,
    })

    state.context = docs
    return docs


def run_agent(state: ConversationState, helper: ServiceHelper, with_tools: bool = False) -> str:
    """Generic agent runner - creates agent, optionally adds MCP tools, runs, and logs."""
    model_name = helper.llms[state.model_id]
    context = "\n\n".join(state.context) if state.context else ""
    files_content = state.files_content or ""
    conversation_history = state.conversation_history or ""

    # Build system content with conversation history, context, and files
    system_content = f"{state.prompt}\n\n{conversation_history}\n\n{context}\n\n{files_content}".strip()

    # Create agent with MCP tools if requested
    if with_tools and helper.tools:
        # Use Pydantic AI's native MCP support via toolsets parameter
        agent = Agent(
            model_name,
            system_prompt=system_content,
            toolsets=[helper.tools]  # Pass MCP server as toolset
        )
    else:
        # Create agent without tools
        agent = Agent(model_name, system_prompt=system_content)

    started_at = timezone.now()
    result = agent.run_sync(state.query)
    ended_at = timezone.now()

    result_text = extract_result_text(result)
    token_counts = estimate_token_counts(state.query, result_text)

    call_id = 'mcp_agent' if with_tools else 'agent'
    state.steps.append({
        'llm_model': state.model_id,
        'system_prompt': state.prompt,
        'call_id': call_id,
        'step_input': state.query,
        'step_output': result_text,
        'started_at': started_at,
        'ended_at': ended_at,
        'token_counts': token_counts,
    })

    state.answer = result_text
    return result_text


def build_conversation_history(conversation):
    """Build a formatted string of previous queries and responses in the conversation."""
    previous_queries = conversation.queries.all().order_by('query_num')

    if not previous_queries.exists():
        return ""

    history_parts = ["### Conversation History:\n"]
    for query in previous_queries:
        history_parts.append(f"USER: {query.raw_query}")
        if query.response:
            history_parts.append(f"ASSISTANT: {query.response}")
        history_parts.append("")  # Empty line for spacing

    return "\n".join(history_parts)


# ==================== Agent Methods ====================

def run_raw_agent(state: ConversationState, helper: ServiceHelper):
    """Direct answering without RAG or tools."""
    state.query = state.raw_query

    # Cache file content if needed
    if state.files and not state.files_content:
        state.files_content = read_and_format_files(state.files)

    return run_agent(state, helper, with_tools=False)


def run_rag_agent(state: ConversationState, helper: ServiceHelper):
    """Answer with document retrieval."""
    state.query = state.raw_query

    # Cache file content if needed
    if state.files and not state.files_content:
        state.files_content = read_and_format_files(state.files)

    retrieve_documents(state, helper)
    return run_agent(state, helper, with_tools=False)


def run_mcp_agent(state: ConversationState, helper: ServiceHelper):
    """Answer with MCP tool calling."""
    state.query = state.raw_query

    # Cache file content if needed
    if state.files and not state.files_content:
        state.files_content = read_and_format_files(state.files)

    return run_agent(state, helper, with_tools=True)


def run_rag_mcp_agent(state: ConversationState, helper: ServiceHelper):
    """Answer with both RAG and MCP tools."""
    state.query = state.raw_query

    # Cache file content if needed
    if state.files and not state.files_content:
        state.files_content = read_and_format_files(state.files)

    retrieve_documents(state, helper)
    return run_agent(state, helper, with_tools=True)


# Agent registry
AGENT_METHODS = {
    'raw': run_raw_agent,
    'rag': run_rag_agent,
    'mcp': run_mcp_agent,
    'rag_mcp': run_rag_mcp_agent
}


# ==================== Token Limit Management ====================

def send_token_limit_email(total_tokens: int, daily_limit: int):
    """Send email notification when daily token limit is exceeded (once per day)."""
    global _last_token_limit_email_sent

    today = timezone.now().date()
    with _email_lock:
        if _last_token_limit_email_sent == today:
            logger.debug("Token limit email already sent today, skipping")
            return
        _last_token_limit_email_sent = today

    admins = getattr(settings, 'ADMINS', [])
    if not admins:
        logger.warning("No ADMINS configured in settings, cannot send token limit email")
        return

    subject = f"Daily Token Limit Exceeded - {timezone.now().strftime('%Y-%m-%d')}"
    message = f"""
Copilot Token Limit Alert
=======================

Copilot has exceeded its daily token usage limit.

Current Usage: {total_tokens:,} tokens
Daily Limit:   {daily_limit:,} tokens
Percentage:    {(total_tokens / daily_limit * 100):.1f}%

Time: {timezone.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
    """.strip()

    try:
        send_mail(
            subject=subject,
            message=message,
            from_email=getattr(settings, 'DEFAULT_FROM_EMAIL', None),
            recipient_list=admins,
            fail_silently=False,
        )
        logger.info(f"Token limit notification email sent to {len(admins)} administrator(s)")
    except Exception as e:
        logger.error(f"Failed to send token limit email: {e}", exc_info=True)


def check_daily_token_limit_exceeded():
    """Check if daily token limit exceeded and send notification if needed."""
    daily_limit = getattr(settings, 'DAILY_TOKEN_LIMIT', 1000000)
    today_start = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)

    result = TokenCount.objects.filter(
        timestamp__gte=today_start,
        token_type=TokenCount.TokenType.TOTAL
    ).aggregate(total=Sum('token_count'))

    total_tokens_today = result['total'] or 0

    if total_tokens_today >= daily_limit:
        logger.warning(f"Daily token limit exceeded: {total_tokens_today}/{daily_limit} tokens")
        send_token_limit_email(total_tokens_today, daily_limit)
        return True

    logger.debug(f"Daily token usage: {total_tokens_today}/{daily_limit} tokens")
    return False


def save_token_counts(step_instance, user, llm_model, token_counts, call_id):
    """Save token count records to database."""
    for token_type, count_value in [
        (TokenCount.TokenType.PROMPT, token_counts.get('prompt_tokens', 0)),
        (TokenCount.TokenType.COMPLETION, token_counts.get('completion_tokens', 0)),
        (TokenCount.TokenType.TOTAL, token_counts.get('total_tokens', 0))
    ]:
        if count_value > 0:
            TokenCount.objects.create(
                step=step_instance,
                user=user,
                llm_model=llm_model,
                token_type=token_type,
                token_count=count_value,
                call_id=str(call_id),
                estimated=token_counts.get('estimated', False)
            )


# ==================== Main Entry Point ====================

def handle_chat_message(user, conversation_id, user_query, model_id=None, method_id=None,
                       system_prompt_id=None, api_key=None, files=None):
    """Handle an incoming chat message and return the response."""

    start_time = timezone.now()
    if user.is_anonymous:
        user = None

    # Log API key for debugging
    logger.info(f"handle_chat_message called with api_key: {api_key is not None}")
    if api_key:
        logger.info(f"  API key length: {len(api_key)}, first 8 chars: {api_key[:8]}")

    # Check token limit
    if check_daily_token_limit_exceeded():
        return None, "I'm sorry. I can't provide you with an answer right now, as I'm too busy answering questions from other users."

    # Get or create conversation
    if conversation_id:
        try:
            conversation = Conversation.objects.get(id=conversation_id)
            if user:
                conversation.user = user
                conversation.save(update_fields=['user'])
        except Conversation.DoesNotExist:
            return None, "Conversation not found or access denied"
    else:
        conversation = Conversation.objects.create(user=user)

    # Get LLM model
    model_id = model_id or settings.DEFAULT_LLM_MODEL
    try:
        llm_model = LlmModel.objects.get(model_id=model_id)
    except LlmModel.DoesNotExist:
        return None, "Requested model id not found"

    # Get method
    method_id = method_id or settings.DEFAULT_LLM_METHOD
    if method_id not in AGENT_METHODS:
        return None, f"Invalid method '{method_id}'. Valid: {list(AGENT_METHODS.keys())}"

    # Get system prompt
    if system_prompt_id:
        system_prompt = SystemPrompt.objects.filter(name=system_prompt_id).order_by('-version').first()
    else:
        system_prompt = SystemPrompt.objects.filter(name="General").order_by('-version').first()

    if not system_prompt:
        return None, "No system prompt found or configured."

    # Build conversation history from previous queries
    conversation_history = build_conversation_history(conversation)

    # Prepare state
    state = ConversationState(
        conversation_id=str(conversation.id),
        model_id=model_id,
        prompt=system_prompt.prompt,
        raw_query=user_query,
        method_id=method_id,
        api_key=api_key,
        files=files or [],
        conversation_history=conversation_history,  # Include conversation history
    )

    # Log state API key
    logger.info(f"ConversationState created with api_key: {state.api_key is not None}")
    if state.api_key:
        logger.info(f"  State API key length: {len(state.api_key)}, first 8 chars: {state.api_key[:8]}")

    # Run agent
    api_key_for_helper = api_key if (method_id in ('mcp', 'rag_mcp') and api_key) else None
    logger.info(f"Creating ServiceHelper with api_key: {api_key_for_helper is not None}")
    if api_key_for_helper:
        logger.info(f"  Helper API key length: {len(api_key_for_helper)}, first 8 chars: {api_key_for_helper[:8]}")

    helper = ServiceHelper.create_instance(api_key=api_key_for_helper)

    try:
        answer = AGENT_METHODS[method_id](state, helper)
    except Exception as e:
        logger.error(f"Error running agent: {e}", exc_info=True)
        return None, f"Error processing request: {str(e)}"

    # Save to database
    end_time = timezone.now()
    query_num = conversation.queries.count() + 1

    query_instance = Query.objects.create(
        conversation=conversation,
        query_num=query_num,
        llm_model=llm_model,
        started_at=start_time,
        ended_at=end_time,
        raw_query=user_query,
        response=answer
    )

    # Save steps and token counts
    for i, step in enumerate(state.steps):
        step_instance = Step.objects.create(
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

        if token_counts := step.get('token_counts'):
            save_token_counts(step_instance, user, llm_model, token_counts, step["call_id"])

    return query_instance, None
