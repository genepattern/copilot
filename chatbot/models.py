import uuid
import os
from django.conf import settings
from django.db import models
from django.contrib.auth.models import User


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    gp_api_key = models.CharField(max_length=255, blank=True, null=True)

    def __str__(self):
        return f"{self.user.username} Profile"


class LlmModel(models.Model):
    """Which LLM model was called"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False, unique=True)
    model_id = models.CharField(max_length=100, unique=True, help_text="Identifier used by the provider (e.g., gpt-4o)")
    provider_id = models.CharField(max_length=100, null=True, blank=True, help_text="Identifier of the provider (e.g., openai)")
    label = models.CharField(max_length=100, help_text="Human-friendly name (e.g., 'OpenAI GPT-4')")
    disabled = models.BooleanField(default=False, help_text="Mark model as disabled")
    max_context_tokens = models.PositiveIntegerField(default=128000, help_text="Maximum context window size in tokens")

    def __str__(self): return self.label


class SystemPrompt(models.Model):
    """Which system prompt was used along with a query"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False, unique=True)
    name = models.CharField(max_length=100, help_text="Human-friendly identifier for the prompt")
    version = models.FloatField(default=1)
    prompt = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('name', 'version')  # Ensure a specific prompt has unique versions
        ordering = ['-created_at']

    def __str__(self): return f"Prompt: {self.name} (v{self.version})"


class Conversation(models.Model):
    """An entry representing one conversation between the user and LLM"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False, unique=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name='conversations')
    label = models.CharField(max_length=200, blank=True, null=True, help_text="Human-friendly label")
    started_at = models.DateTimeField(auto_now_add=True)

    class Meta: ordering = ['-started_at']

    def __str__(self): return f"Conversation {self.id} for {self.user.username if self.user else 'Anonymous'}"


class Query(models.Model):
    """One prompt-response between the user and LLM within a conversation"""

    class Rating(models.IntegerChoices):
        """Thumbs up or down rating for a response"""
        THUMBS_DOWN = -1, 'Thumbs Down'
        NO_RATING = 0, 'No Rating'
        THUMBS_UP = 1, 'Thumbs Up'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False, unique=True)
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='queries')
    query_num = models.PositiveIntegerField(help_text="Order of the query within the conversation")
    llm_model = models.ForeignKey(LlmModel, on_delete=models.SET_NULL, null=True, blank=True, related_name='queries')
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, help_text="Datetime of the final response")
    raw_query = models.TextField(help_text="The user's raw input")
    response = models.TextField(null=True, blank=True, help_text="The LLM's final response")
    rating = models.IntegerField(choices=Rating.choices, default=Rating.NO_RATING, null=True, blank=True)

    class Meta:
        ordering = ['started_at']  # Order queries within a conversation chronologically
        verbose_name_plural = "Queries"  # Correct pluralization

    def __str__(self):
        return f"Query {self.query_num} of Conversation {self.conversation.label if self.conversation.label else self.conversation.id}"


class Step(models.Model):
    """Represents one step in the chain used to generate a response for a query"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False, unique=True)
    query = models.ForeignKey(Query, on_delete=models.CASCADE, related_name='steps')
    step_num = models.PositiveIntegerField(help_text="Step number to generate the query")

    llm_model = models.ForeignKey(LlmModel, on_delete=models.SET_NULL, null=True, blank=True, related_name='calls', help_text="LLM used in this specific step")
    system_prompt = models.ForeignKey(SystemPrompt, on_delete=models.SET_NULL, null=True, blank=True, related_name='calls')
    call_id = models.CharField(max_length=100, help_text="Used to identify the step in the graph")

    step_input = models.TextField(null=True, blank=True, help_text="LLM input being passed to this step")
    step_output = models.TextField(null=True, blank=True, help_text="LLM output coming from this step")

    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, help_text="Datetime of the final response")

    class Meta:
        unique_together = ('query', 'step_num')  # Ensure step numbers are unique per query
        ordering = ['-query', 'step_num']  # Order steps within a query chronologically

    def __str__(self): return f"Step {self.step_num} of Query {self.query.id}"


class TokenCount(models.Model):
    """Records token usage for LLM calls"""

    class TokenType(models.TextChoices):
        PROMPT = 'prompt', 'Prompt (Input)'
        COMPLETION = 'completion', 'Completion (Output)'
        TOTAL = 'total', 'Total'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False, unique=True)
    step = models.ForeignKey(Step, on_delete=models.CASCADE, null=True, blank=True, related_name='token_counts', help_text="Associated step if applicable")
    user = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name='token_usage', help_text="User who initiated the request (null for anonymous)")
    llm_model = models.ForeignKey(LlmModel, on_delete=models.SET_NULL, null=True, blank=True, related_name='token_usage')

    token_type = models.CharField(max_length=20, choices=TokenType.choices, help_text="Type of tokens (prompt/completion/total)")
    token_count = models.PositiveIntegerField(help_text="Number of tokens used")

    timestamp = models.DateTimeField(auto_now_add=True, help_text="When this token usage occurred")

    # Optional metadata
    call_id = models.CharField(max_length=100, null=True, blank=True, help_text="Identifier for the graph node that made this call")
    estimated = models.BooleanField(default=False, help_text="Whether this count is estimated (for models without native token tracking)")

    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['user', 'timestamp']),
            models.Index(fields=['llm_model', 'timestamp']),
            models.Index(fields=['token_type']),
        ]

    def __str__(self):
        user_str = self.user.username if self.user else 'Anonymous'
        model_str = self.llm_model.model_id if self.llm_model else 'Unknown'
        return f"{self.token_count} {self.token_type} tokens - {model_str} - {user_str}"

    def get_estimated_cost(self):
        """
        Calculate the estimated cost for this token usage.
        Returns the cost in dollars based on token type and count.
        """
        INPUT_TOKEN_COST_PER_1000 = float(os.getenv('INPUT_TOKEN_COST_PER_1000', '0.001'))
        OUTPUT_TOKEN_COST_PER_1000 = float(os.getenv('OUTPUT_TOKEN_COST_PER_1000', '0.005'))

        if self.token_type == self.TokenType.PROMPT:
            return (self.token_count / 1000) * INPUT_TOKEN_COST_PER_1000
        elif self.token_type == self.TokenType.COMPLETION:
            return (self.token_count / 1000) * OUTPUT_TOKEN_COST_PER_1000
        elif self.token_type == self.TokenType.TOTAL:
            # For total, we can't distinguish between input/output, so use average
            avg_cost = (INPUT_TOKEN_COST_PER_1000 + OUTPUT_TOKEN_COST_PER_1000) / 2
            return (self.token_count / 1000) * avg_cost
        return 0.0
