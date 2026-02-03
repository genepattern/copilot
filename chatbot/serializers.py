from bs4 import BeautifulSoup
from markdown_it import MarkdownIt
from rest_framework import serializers
from .models import Conversation, Query, LlmModel, Step


class LlmModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = LlmModel
        fields = ['model_id', 'provider_id', 'label', 'disabled']


class StepSerializer(serializers.ModelSerializer):
    class Meta:
        model = Step
        fields = ['id', 'query', 'step_num', 'llm_model', 'system_prompt', 'call_id', 'step_input',
                  'step_output', 'started_at', 'ended_at']  # 'llm_model_details' -- Include FK and nested details
        read_only_fields = ['id', 'step_num', 'started_at', 'ended_at']


class QuerySerializer(serializers.ModelSerializer):
    steps = StepSerializer(many=True, read_only=True)
    model = LlmModelSerializer(source='llm_model', read_only=True)
    conversation = serializers.UUIDField(source='conversation.id', read_only=True)  # UUID of the conversation
    rating_label = serializers.CharField(source='get_rating_display', read_only=True)  # Human-readable rating
    query = serializers.CharField(source='raw_query', read_only=True)  # User's query
    response = serializers.SerializerMethodField()  # LLM's response, encoded as HTML or Markdown
    estimated_cost = serializers.SerializerMethodField()  # Estimated cost for this query

    class Meta:
        model = Query
        fields = [
            'id', 'conversation', 'query_num', 'started_at', 'ended_at',
            'rating', 'rating_label', 'query', 'response', 'model', 'steps', 'estimated_cost'
        ]
        read_only_fields = ['id', 'started_at', 'ended_at', 'steps', 'llm_model']

    @staticmethod
    def markdown_to_html(markdown_text):
        if not markdown_text:
            return ""
        parser = MarkdownIt("gfm-like")
        html = parser.render(markdown_text)
        soup = BeautifulSoup(html, 'html.parser')

        # Add Bootstrap classes to tables
        for table in soup.find_all('table'):
            table['class'] = table.get('class', []) + ['table', 'table-striped']

        for a in soup.find_all('a', href=True):
            a['target'] = '_blank'
        return str(soup)

    def get_response(self, obj):
        # Default to plain text unless 'html' is explicitly requested in context
        if self.context.get('html', False):
            return self.markdown_to_html(obj.response)
        else:
            return obj.response

    def get_estimated_cost(self, obj):
        """Calculate estimated cost for all token counts in this query's steps."""
        total_cost = 0.0
        for step in obj.steps.all():
            for token_count in step.token_counts.all():
                total_cost += token_count.get_estimated_cost()
        return round(total_cost, 6)


class ConversationListSerializer(serializers.ModelSerializer):
    title = serializers.SerializerMethodField()

    class Meta:
        model = Conversation
        fields = ['id', 'title', 'started_at']
        read_only_fields = ['id', 'started_at']

    def get_title(self, obj):
        # Prefer explicit label; fallback to first query's text; otherwise a default placeholder
        if getattr(obj, 'label', None):
            return obj.label
        first_query = None
        try:
            # Queries are ordered by started_at ascending per model Meta
            first_query = obj.queries.first()
        except Exception:
            first_query = None
        if first_query and getattr(first_query, 'raw_query', None):
            return first_query.raw_query
        return "New conversation"


class ConversationSerializer(serializers.ModelSerializer):
    queries = QuerySerializer(many=True, read_only=True)

    class Meta:
        model = Conversation
        fields = ['id', 'user', 'label', 'started_at', 'queries']
        read_only_fields = ['id', 'user', 'started_at', 'queries']


class ChatInputSerializer(serializers.Serializer):
    """Serializer for receiving a new chat message"""

    conversation_id = serializers.UUIDField(required=False, allow_null=True, help_text="Omit to start a new conversation")
    query = serializers.CharField(max_length=10000, help_text="The user's query")
    model_id = serializers.CharField(max_length=100, required=False, allow_null=True, help_text="Specific model id to use")
    method_id = serializers.CharField(max_length=100, required=False, allow_null=True, help_text="Specific method to use")
    api_key = serializers.CharField(max_length=100, required=False, allow_null=True, help_text="GenePattern API key for authenticated MCP calls")
    html = serializers.BooleanField(default=False, help_text="Return HTML response, otherwise return Markdown")
    files = serializers.ListField(
        child=serializers.FileField(),
        required=False,
        allow_empty=True,
        help_text="List of files to attach to the query"
    )

    # Future Use?: Allow specifying a system prompt version or ID
    # system_prompt_id = serializers.IntegerField(required=False, allow_null=True)

    def validate_model_id(self, value):
        """Check if the provided model_id exists."""
        if value and not LlmModel.objects.filter(model_id=value).exists():
            raise serializers.ValidationError(f"LLM Model with id '{value}' not found.")
        return value

    def validate_method_id(self, value):
        """Normalize and validate method_id"""
        if value is None: return value
        normalized = value.strip().lower()
        allowed = {'rag', 'mcp', 'raw', 'rag_mcp'}
        if normalized not in allowed:
            raise serializers.ValidationError(f"Invalid method_id '{value}'. Must be one of {sorted(allowed)}.")
        return normalized


class QueryRatingSerializer(serializers.Serializer):
    """Serializer for updating the rating of a query."""
    rating = serializers.ChoiceField(choices=Query.Rating.choices)
