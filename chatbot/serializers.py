from bs4 import BeautifulSoup
from markdown import markdown
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

    class Meta:
        model = Query
        fields = [
            'id', 'conversation', 'query_num', 'started_at', 'ended_at',
            'rating', 'rating_label', 'query', 'response', 'model', 'steps'
        ]
        read_only_fields = ['id', 'started_at', 'ended_at', 'steps', 'llm_model']

    @staticmethod
    def markdown_to_html(markdown_text):
        html = markdown(markdown_text, extensions=['extra', 'smarty'])
        soup = BeautifulSoup(html, 'html.parser')
        for a in soup.find_all('a', href=True): a['target'] = '_blank'
        return str(soup)

    def get_response(self, obj):
        if self.context['html']: return self.markdown_to_html(obj.response)
        else: return obj.response


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
    html = serializers.BooleanField(default=False, help_text="Return HTML response, otherwise return Markdown")

    # Future Use?: Allow specifying a system prompt version or ID
    # system_prompt_id = serializers.IntegerField(required=False, allow_null=True)

    def validate_model_id(self, value):
        """Check if the provided model_id exists."""
        if value and not LlmModel.objects.filter(model_id=value).exists():
            raise serializers.ValidationError(f"LLM Model with id '{value}' not found.")
        return value


class QueryRatingSerializer(serializers.Serializer):
    """Serializer for updating the rating of a query."""
    rating = serializers.ChoiceField(choices=Query.Rating.choices)
